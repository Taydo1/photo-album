using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using OpenCvSharp;
using Razorvine.Pickle;
using TorchSharp;
using TorchSharp.Modules;
using TorchSharp.PyBridge;
using static TorchSharp.torch;

namespace PhotoApp2.Services
{
    public class OnnxContentClassifier : IDisposable
    {
        private torch.nn.Module<Tensor, Tensor>? _model;
        private readonly object _inferLock = new object();
        private bool _isGpuEnabled;
        private int _inputWidth = 224;
        private int _inputHeight = 224;

        public bool IsGpuEnabled => _isGpuEnabled;
        public bool IsInitialized => _model != null;

        public async Task InitializeAsync(string? modelPath = null)
        {
            if (_model != null) return;

            await Task.Run(async () =>
            {
                try
                {
                    var appDataDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "PhotoApp2", "Models");
                    Directory.CreateDirectory(appDataDir);

                    string? rawCheckpointPath = modelPath;
                    if (string.IsNullOrEmpty(rawCheckpointPath))
                    {
                        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
                        rawCheckpointPath = Path.Combine(baseDir, "Assets", "Models", "resnet18_places365.pth.tar");
                    }

                    if (!File.Exists(rawCheckpointPath))
                    {
                        var downloadPath = Path.Combine(appDataDir, "resnet18_places365.pth.tar");
                        if (!File.Exists(downloadPath))
                        {
                            Debug.WriteLine("Downloading Places365 ResNet18 model from MIT CSAIL...");
                            try
                            {
                                using var client = new HttpClient();
                                client.Timeout = TimeSpan.FromMinutes(5);
                                var data = await client.GetByteArrayAsync("http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar");
                                await File.WriteAllBytesAsync(downloadPath, data);
                                Debug.WriteLine($"Downloaded Places365 model ({data.Length / 1024 / 1024} MB)");
                            }
                            catch (Exception downloadEx)
                            {
                                Debug.WriteLine($"Auto-download of Places365 model failed: {downloadEx.Message}");
                            }
                        }
                        if (File.Exists(downloadPath))
                            rawCheckpointPath = downloadPath;
                    }

                    // Instantiate TorchVision ResNet18 with exactly 365 output classes
                    var resnet = TorchSharp.torchvision.models.resnet18(num_classes: 365);

                    if (File.Exists(rawCheckpointPath))
                    {
                        Debug.WriteLine($"Loading Places365 weights from legacy PyTorch checkpoint: {rawCheckpointPath}...");
                        LoadLegacyPthTarWeights(resnet, rawCheckpointPath!);
                        Debug.WriteLine("Successfully loaded Places365 weights (365 output classes).");
                    }
                    else
                    {
                        Debug.WriteLine("WARNING: No Places365 weights available. Scene classification will be unreliable.");
                    }

                    resnet.eval();
                    _model = resnet;
                    _isGpuEnabled = false;
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error initializing OnnxContentClassifier: {ex.Message}");
                    Console.WriteLine($"Error initializing OnnxContentClassifier: {ex.Message}");
                }
            });
        }

        /// <summary>
        /// Directly reads MIT CSAIL's legacy PyTorch (.pth.tar) binary stream checkpoint and populates ResNet18 tensors in-place.
        /// Handles legacy sequential pickle streams, persistent storage references, and strips "module." DataParallel prefixes.
        /// </summary>
        private static void LoadLegacyPthTarWeights(TorchSharp.Modules.ResNet resnet, string checkpointPath)
        {
            Razorvine.Pickle.Unpickler.registerConstructor("torch._utils", "_rebuild_tensor", new PyTorchObjectConstructor());
            Razorvine.Pickle.Unpickler.registerConstructor("torch._utils", "_rebuild_tensor_v2", new PyTorchObjectConstructor());
            Razorvine.Pickle.Unpickler.registerConstructor("torch._utils", "_rebuild_parameter", new PyTorchObjectConstructor());
            Razorvine.Pickle.Unpickler.registerConstructor("collections", "OrderedDict", new PyTorchObjectConstructor());

            var modelStateDict = resnet.state_dict();
            using var fs = File.OpenRead(checkpointPath);
            var unpickler = new LegacyPyTorchUnpickler();

            // Read legacy sequential header streams
            unpickler.load(fs); // Stream 1: Magic number (0x1950a86a20f9469c)
            unpickler.load(fs); // Stream 2: Protocol version (1001)
            unpickler.load(fs); // Stream 3: System metadata table
            var obj4 = unpickler.load(fs); // Stream 4: Checkpoint dictionary
            var storageListObj = unpickler.load(fs); // Stream 5: Storage IDs order list

            if (obj4 is System.Collections.Hashtable ht4 && ht4.ContainsKey("state_dict") &&
                ht4["state_dict"] is System.Collections.Hashtable sdHt &&
                storageListObj is System.Collections.ArrayList storages)
            {
                var storageMap = new Dictionary<string, float[]>();
                using var reader = new BinaryReader(fs);
                foreach (object? storageIdObj in storages)
                {
                    if (storageIdObj == null) continue;
                    string sId = storageIdObj.ToString()!;
                    long numEl = reader.ReadInt64();
                    byte[] bytes = reader.ReadBytes((int)(numEl * 4));
                    float[] floats = new float[numEl];
                    Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
                    storageMap[sId] = floats;
                }

                int loadedCount = 0;
                using (TorchSharp.torch.no_grad())
                {
                    foreach (object keyObj in sdHt.Keys)
                    {
                        string rawKey = keyObj.ToString()!;
                        string cleanKey = rawKey.StartsWith("module.") ? rawKey.Substring(7) : rawKey;

                        if (sdHt[keyObj] is TensorRef tr && tr.Args != null && tr.Args.Length >= 3)
                        {
                            if (tr.Args[0] is StorageRef sr && sr.Data != null && sr.Data.Length > 2)
                            {
                                string storageId = sr.Data[2]?.ToString() ?? "";
                                long[] shape = Array.Empty<long>();
                                if (tr.Args[2] is System.Collections.ArrayList shapeAl)
                                {
                                    shape = shapeAl.Cast<object>().Select(x => Convert.ToInt64(x)).ToArray();
                                }
                                else if (tr.Args[2] is object[] shapeArr)
                                {
                                    shape = shapeArr.Select(x => Convert.ToInt64(x)).ToArray();
                                }

                                if (storageMap.TryGetValue(storageId, out var floatData) && modelStateDict.TryGetValue(cleanKey, out var targetTensor))
                                {
                                    using var srcTensor = TorchSharp.torch.tensor(floatData, shape);
                                    targetTensor.copy_(srcTensor);
                                    loadedCount++;
                                }
                            }
                        }
                    }
                }
                Debug.WriteLine($"Loaded {loadedCount} tensors into ResNet18 from Places365 checkpoint.");
            }
        }

        private class LegacyPyTorchUnpickler : Razorvine.Pickle.Unpickler
        {
            protected override object persistentLoad(object pid)
            {
                if (pid is object[] arr) return new StorageRef { Data = arr };
                return pid!;
            }
        }

        private class StorageRef
        {
            public object[]? Data { get; set; }
        }

        private class TensorRef
        {
            public object[]? Args { get; set; }
        }

        private class PyTorchObjectConstructor : Razorvine.Pickle.IObjectConstructor
        {
            public object construct(object[] args)
            {
                if (args != null && args.Length == 1 && args[0] is System.Collections.ArrayList al)
                {
                    var dict = new System.Collections.Hashtable();
                    foreach (object item in al)
                    {
                        if (item is object[] pair && pair.Length == 2 && pair[0] != null)
                            dict[pair[0]!.ToString()!] = pair[1];
                        else if (item is System.Collections.ArrayList pairList && pairList.Count == 2 && pairList[0] != null)
                            dict[pairList[0]!.ToString()!] = pairList[1];
                    }
                    if (dict.Count > 0) return dict;
                }
                return new TensorRef { Args = args };
            }
        }

        public Task<(List<string> Tags, double Confidence, float[] FeatureVector)> ClassifySceneAsync(Mat inputMat)
            => Task.Run(() => ClassifyScene(inputMat));

        public (List<string> Tags, double Confidence, float[] FeatureVector) ClassifyScene(Mat inputMat)
        {
            if (_model == null || inputMat.Empty())
                return (new List<string> { "Other" }, 0.0, Array.Empty<float>());

            try
            {
                using var resized = new Mat();
                Cv2.Resize(inputMat, resized, new OpenCvSharp.Size(_inputWidth, _inputHeight), 0, 0, InterpolationFlags.Linear);

                using var rgb = new Mat();
                if (resized.Channels() == 1)
                {
                    Cv2.CvtColor(resized, rgb, ColorConversionCodes.GRAY2RGB);
                }
                else if (resized.Channels() == 3)
                {
                    Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);
                }
                else if (resized.Channels() == 4)
                {
                    Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGRA2RGB);
                }
                else
                {
                    resized.CopyTo(rgb);
                }

                // Prepare input float array [1, 3, 224, 224] with standard ImageNet normalization
                float[] inputBuffer = new float[1 * 3 * _inputHeight * _inputWidth];
                const float meanR = 0.485f, meanG = 0.456f, meanB = 0.406f;
                const float stdR = 0.229f, stdG = 0.224f, stdB = 0.225f;

                unsafe
                {
                    byte* ptr = (byte*)rgb.Data;
                    int stride = (int)rgb.Step();
                    int planeSize = _inputHeight * _inputWidth;

                    for (int y = 0; y < _inputHeight; y++)
                    {
                        byte* row = ptr + (y * stride);
                        for (int x = 0; x < _inputWidth; x++)
                        {
                            int idx = x * 3;
                            int pixelIdx = y * _inputWidth + x;
                            inputBuffer[0 * planeSize + pixelIdx] = ((row[idx] / 255.0f) - meanR) / stdR;
                            inputBuffer[1 * planeSize + pixelIdx] = ((row[idx + 1] / 255.0f) - meanG) / stdG;
                            inputBuffer[2 * planeSize + pixelIdx] = ((row[idx + 2] / 255.0f) - meanB) / stdB;
                        }
                    }
                }

                float[] logits;
                lock (_inferLock)
                {
                    using (no_grad())
                    {
                        using var inputTensor = torch.tensor(inputBuffer, new long[] { 1, 3, _inputHeight, _inputWidth });
                        using var outputTensor = _model.call(inputTensor);
                        logits = outputTensor.data<float>().ToArray();
                    }
                }

                float[] softmaxProbs = CalculateSoftmax(logits);
                return ProcessClassificationResults(softmaxProbs);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error during TorchSharp scene classification: {ex.Message}");
                return (new List<string> { "Other" }, 0.0, Array.Empty<float>());
            }
        }

        private static float[] CalculateSoftmax(float[] logits)
        {
            if (logits == null || logits.Length == 0) return Array.Empty<float>();

            float maxLogit = logits.Max();
            float[] exp = new float[logits.Length];
            double sumExp = 0.0;

            for (int i = 0; i < logits.Length; i++)
            {
                exp[i] = MathF.Exp(logits[i] - maxLogit);
                sumExp += exp[i];
            }

            float[] softmax = new float[logits.Length];
            if (sumExp > 0)
            {
                for (int i = 0; i < logits.Length; i++)
                {
                    softmax[i] = (float)(exp[i] / sumExp);
                }
            }
            return softmax;
        }

        public static double CalculateCosineSimilarity(float[]? vectorA, float[]? vectorB)
        {
            if (vectorA == null || vectorB == null || vectorA.Length == 0 || vectorB.Length == 0)
                return 0.0;

            int minLength = Math.Min(vectorA.Length, vectorB.Length);
            double dotProduct = 0.0;
            double normA = 0.0;
            double normB = 0.0;

            for (int i = 0; i < minLength; i++)
            {
                double a = vectorA[i];
                double b = vectorB[i];
                dotProduct += a * b;
                normA += a * a;
                normB += b * b;
            }

            if (normA <= 0.0 || normB <= 0.0) return 0.0;
            return dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
        }

        private static (List<string> Tags, double Confidence, float[] FeatureVector) ProcessClassificationResults(float[] probs)
        {
            if (probs == null || probs.Length == 0)
                return (new List<string> { "Other" }, 0.0, Array.Empty<float>());

            int topIndex = 0;
            float maxProb = probs[0];
            for (int i = 1; i < probs.Length; i++)
            {
                if (probs[i] > maxProb)
                {
                    maxProb = probs[i];
                    topIndex = i;
                }
            }

            double confidence = Math.Clamp(maxProb, 0.0, 1.0);

            // Sum probability of all classes mapping to the same primary tag
            var tagProbs = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < probs.Length && i < CategoryPrimaryKinds.Length; i++)
            {
                string tag = CategoryPrimaryKinds[i];
                if (!string.IsNullOrEmpty(tag) && tag != "Other")
                {
                    if (tagProbs.TryGetValue(tag, out float currentProb))
                    {
                        tagProbs[tag] = currentProb + probs[i];
                    }
                    else
                    {
                        tagProbs[tag] = probs[i];
                    }
                }
            }

            var tagsList = new List<string>();

            // Filter primary kind tags with total probability >= 0.15f, ordered descending by total probability
            var validTagPairs = tagProbs
                .Where(kvp => kvp.Value >= 0.15f)
                .OrderByDescending(kvp => kvp.Value)
                .ToList();

            if (validTagPairs.Any())
            {
                foreach (var pair in validTagPairs)
                {
                    tagsList.Add(pair.Key);
                }
            }
            else
            {
                // Fallback to top index's primary tag if no tag reached 0.15f
                string topTag = MapIndexToPrimaryKind(topIndex);
                if (!string.IsNullOrEmpty(topTag) && topTag != "Other")
                {
                    tagsList.Add(topTag);
                }
            }

            if (!tagsList.Any())
            {
                tagsList.Add("Other");
            }

            return (tagsList, confidence, probs);
        }

        private static string MapIndexToPrimaryKind(int index)
        {
            if (index >= 0 && index < CategoryPrimaryKinds.Length)
            {
                return CategoryPrimaryKinds[index];
            }
            return "Other";
        }

        private static readonly string[] CategoryPrimaryKinds = new string[]
        {
            // 0-29
            "Urban & Travel", "Urban & Travel", "Urban & Travel", "Home & Indoors", "Urban & Travel",
            "Leisure & Recreation", "Leisure & Recreation", "Leisure & Recreation", "Architecture", "Leisure & Recreation",
            "Architecture", "Leisure & Recreation", "Architecture", "Architecture", "Culture & Education",
            "Leisure & Recreation", "Leisure & Recreation", "Leisure & Recreation", "Work & Industry", "Culture & Education",
            "Culture & Education", "Culture & Education", "Culture & Education", "Work & Industry", "Leisure & Recreation",
            "Architecture", "Home & Indoors", "Culture & Education", "Work & Industry", "Work & Industry",

            // 30-59
            "Landscape", "Food & Dining", "Architecture", "Home & Indoors", "Leisure & Recreation",
            "Leisure & Recreation", "Landscape", "Work & Industry", "Food & Dining", "Food & Dining",
            "Architecture", "Architecture", "Leisure & Recreation", "Home & Indoors", "Leisure & Recreation",
            "Home & Indoors", "Urban & Travel", "Urban & Travel", "Landscape", "Architecture",
            "Work & Industry", "Home & Indoors", "Home & Indoors", "Food & Dining", "Food & Dining",
            "Urban & Travel", "Work & Industry", "Urban & Travel", "Urban & Travel", "Architecture",

            // 60-89
            "Culture & Education", "Work & Industry", "Leisure & Recreation", "Home & Indoors", "Leisure & Recreation",
            "Leisure & Recreation", "Architecture", "Architecture", "Leisure & Recreation", "Architecture",
            "Urban & Travel", "Urban & Travel", "Food & Dining", "Landscape", "Architecture",
            "Food & Dining", "Leisure & Recreation", "Culture & Education", "Landscape", "Urban & Travel",
            "Food & Dining", "Landscape", "Urban & Travel", "Leisure & Recreation", "Architecture",
            "Architecture", "Architecture", "Architecture", "Work & Industry", "Home & Indoors",

            // 90-119
            "Architecture", "Architecture", "Culture & Education", "Work & Industry", "Landscape",
            "Home & Indoors", "Work & Industry", "Landscape", "Urban & Travel", "Food & Dining",
            "Work & Industry", "Work & Industry", "Work & Industry", "Work & Industry", "Landscape",
            "Work & Industry", "Home & Indoors", "Architecture", "Culture & Education", "Architecture",
            "Landscape", "Landscape", "Urban & Travel", "Architecture", "Food & Dining",
            "Work & Industry", "Landscape", "Landscape", "Landscape", "Food & Dining",

            // 120-149
            "Food & Dining", "Food & Dining", "Leisure & Recreation", "Architecture", "Home & Indoors",
            "Urban & Travel", "Home & Indoors", "Urban & Travel", "Work & Industry", "Home & Indoors",
            "Home & Indoors", "Architecture", "Work & Industry", "Work & Industry", "Home & Indoors",
            "Urban & Travel", "Work & Industry", "Work & Industry", "Work & Industry", "Food & Dining",
            "Landscape", "Landscape", "Landscape", "Architecture", "Urban & Travel",
            "Landscape", "Urban & Travel", "Work & Industry", "Food & Dining", "Leisure & Recreation",

            // 150-179
            "Landscape", "Landscape", "Landscape", "Leisure & Recreation", "Architecture",
            "Home & Indoors", "Home & Indoors", "Urban & Travel", "Urban & Travel", "Architecture",
            "Work & Industry", "Work & Industry", "Work & Industry", "Landscape", "Leisure & Recreation",
            "Work & Industry", "Work & Industry", "Landscape", "Leisure & Recreation", "Work & Industry",
            "Work & Industry", "Urban & Travel", "Work & Industry", "Landscape", "Urban & Travel",
            "Urban & Travel", "Home & Indoors", "Home & Indoors", "Work & Industry", "Home & Indoors",

            // 180-209
            "Landscape", "Urban & Travel", "Home & Indoors", "Architecture", "Architecture",
            "Food & Dining", "Landscape", "Landscape", "Leisure & Recreation", "Leisure & Recreation",
            "Landscape", "Architecture", "Work & Industry", "Urban & Travel", "Landscape",
            "Home & Indoors", "Work & Industry", "Leisure & Recreation", "Work & Industry", "Work & Industry",
            "Urban & Travel", "Work & Industry", "Culture & Education", "Home & Indoors", "Landscape",
            "Landscape", "Work & Industry", "Urban & Travel", "Work & Industry", "Landscape",

            // 210-239
            "Culture & Education", "Culture & Education", "Culture & Education", "Culture & Education", "Architecture",
            "Home & Indoors", "Urban & Travel", "Home & Indoors", "Urban & Travel", "Home & Indoors",
            "Architecture", "Architecture", "Urban & Travel", "Urban & Travel", "Landscape",
            "Leisure & Recreation", "Architecture", "Urban & Travel", "Architecture", "Landscape",
            "Architecture", "Urban & Travel", "Landscape", "Landscape", "Landscape",
            "Leisure & Recreation", "Culture & Education", "Culture & Education", "Leisure & Recreation", "Culture & Education",

            // 240-269
            "Home & Indoors", "Work & Industry", "Architecture", "Landscape", "Work & Industry",
            "Work & Industry", "Work & Industry", "Work & Industry", "Work & Industry", "Landscape",
            "Leisure & Recreation", "Architecture", "Architecture", "Home & Indoors", "Leisure & Recreation",
            "Urban & Travel", "Urban & Travel", "Urban & Travel", "Landscape", "Home & Indoors",
            "Architecture", "Work & Industry", "Work & Industry", "Urban & Travel", "Work & Industry",
            "Food & Dining", "Architecture", "Food & Dining", "Leisure & Recreation", "Home & Indoors",

            // 270-299
            "Urban & Travel", "Landscape", "Home & Indoors", "Urban & Travel", "Food & Dining",
            "Leisure & Recreation", "Leisure & Recreation", "Leisure & Recreation", "Urban & Travel", "Landscape",
            "Home & Indoors", "Home & Indoors", "Work & Industry", "Urban & Travel", "Food & Dining",
            "Food & Dining", "Food & Dining", "Landscape", "Landscape", "Landscape",
            "Architecture", "Architecture", "Architecture", "Urban & Travel", "Leisure & Recreation",
            "Work & Industry", "Culture & Education", "Culture & Education", "Work & Industry", "Architecture",

            // 300-329
            "Work & Industry", "Work & Industry", "Work & Industry", "Home & Indoors", "Leisure & Recreation",
            "Leisure & Recreation", "Landscape", "Architecture", "Urban & Travel", "Landscape",
            "Leisure & Recreation", "Work & Industry", "Leisure & Recreation", "Leisure & Recreation", "Leisure & Recreation",
            "Leisure & Recreation", "Leisure & Recreation", "Architecture", "Home & Indoors", "Urban & Travel",
            "Urban & Travel", "Work & Industry", "Food & Dining", "Landscape", "Landscape",
            "Leisure & Recreation", "Leisure & Recreation", "Architecture", "Home & Indoors", "Work & Industry",

            // 330-359
            "Architecture", "Architecture", "Urban & Travel", "Leisure & Recreation", "Architecture",
            "Work & Industry", "Urban & Travel", "Urban & Travel", "Landscape", "Architecture",
            "Landscape", "Landscape", "Landscape", "Home & Indoors", "Landscape",
            "Landscape", "Work & Industry", "Architecture", "Urban & Travel", "Landscape",
            "Landscape", "Leisure & Recreation", "Home & Indoors", "Leisure & Recreation", "Architecture",
            "Landscape", "Landscape", "Landscape", "Food & Dining", "Landscape",

            // 360-364
            "Work & Industry", "Architecture", "Home & Indoors", "Urban & Travel", "Leisure & Recreation"
        };

        public void Dispose()
        {
            _model?.Dispose();
            _model = null;
            GC.SuppressFinalize(this);
        }

        private static readonly string[] Places365Categories = new string[]
        {
            "/a/airfield", "/a/airplane_cabin", "/a/airport_terminal", "/a/alcove", "/a/alley",
            "/a/amphitheater", "/a/amusement_arcade", "/a/amusement_park", "/a/apartment_building/outdoor", "/a/aquarium",
            "/a/aqueduct", "/a/arcade", "/a/arch", "/a/archaelogical_excavation", "/a/archive",
            "/a/arena/hockey", "/a/arena/performance", "/a/arena/rodeo", "/a/army_base", "/a/art_gallery",
            "/a/art_school", "/a/art_studio", "/a/artists_loft", "/a/assembly_line", "/a/athletic_field/outdoor",
            "/a/atrium/public", "/a/attic", "/a/auditorium", "/a/auto_factory", "/a/auto_showroom",
            "/b/badlands", "/b/bakery/shop", "/b/balcony/exterior", "/b/balcony/interior", "/b/ball_pit",
            "/b/ballroom", "/b/bamboo_forest", "/b/bank_vault", "/b/banquet_hall", "/b/bar",
            "/b/barn", "/b/barndoor", "/b/baseball_field", "/b/basement", "/b/basketball_court/indoor",
            "/b/bathroom", "/b/bazaar/indoor", "/b/bazaar/outdoor", "/b/beach", "/b/beach_house",
            "/b/beauty_salon", "/b/bedchamber", "/b/bedroom", "/b/beer_garden", "/b/beer_hall",
            "/b/berth", "/b/biology_laboratory", "/b/boardwalk", "/b/boat_deck", "/b/boathouse",
            "/b/bookstore", "/b/booth/indoor", "/b/botanical_garden", "/b/bow_window/indoor", "/b/bowling_alley",
            "/b/boxing_ring", "/b/bridge", "/b/building_facade", "/b/bullring", "/b/burial_chamber",
            "/b/bus_interior", "/b/bus_station/indoor", "/b/butchers_shop", "/b/butte", "/c/cabin/outdoor",
            "/c/cafeteria", "/c/campsite", "/c/campus", "/c/canal/natural", "/c/canal/urban",
            "/c/candy_store", "/c/canyon", "/c/car_interior", "/c/carrousel", "/c/castle",
            "/c/catacomb", "/c/cemetery", "/c/chalet", "/c/chemistry_lab", "/c/childs_room",
            "/c/church/indoor", "/c/church/outdoor", "/c/classroom", "/c/clean_room", "/c/cliff",
            "/c/closet", "/c/clothing_store", "/c/coast", "/c/cockpit", "/c/coffee_shop",
            "/c/computer_room", "/c/conference_center", "/c/conference_room", "/c/construction_site", "/c/corn_field",
            "/c/corral", "/c/corridor", "/c/cottage", "/c/courthouse", "/c/courtyard",
            "/c/creek", "/c/crevasse", "/c/crosswalk", "/d/dam", "/d/delicatessen",
            "/d/department_store", "/d/desert/sand", "/d/desert/vegetation", "/d/desert_road", "/d/diner/outdoor",
            "/d/dining_hall", "/d/dining_room", "/d/discotheque", "/d/doorway/outdoor", "/d/dorm_room",
            "/d/downtown", "/d/dressing_room", "/d/driveway", "/d/drugstore", "/e/elevator/door",
            "/e/elevator_lobby", "/e/elevator_shaft", "/e/embassy", "/e/engine_room", "/e/entrance_hall",
            "/e/escalator/indoor", "/e/excavation", "/f/fabric_store", "/f/farm", "/f/fastfood_restaurant",
            "/f/field/cultivated", "/f/field/wild", "/f/field_road", "/f/fire_escape", "/f/fire_station",
            "/f/fishpond", "/f/flea_market/indoor", "/f/florist_shop/indoor", "/f/food_court", "/f/football_field",
            "/f/forest/broadleaf", "/f/forest_path", "/f/forest_road", "/f/formal_garden", "/f/fountain",
            "/g/galley", "/g/garage/indoor", "/g/garage/outdoor", "/g/gas_station", "/g/gazebo/exterior",
            "/g/general_store/indoor", "/g/general_store/outdoor", "/g/gift_shop", "/g/glacier", "/g/golf_course",
            "/g/greenhouse/indoor", "/g/greenhouse/outdoor", "/g/grotto", "/g/gymnasium/indoor", "/h/hangar/indoor",
            "/h/hangar/outdoor", "/h/harbor", "/h/hardware_store", "/h/hayfield", "/h/heliport",
            "/h/highway", "/h/home_office", "/h/home_theater", "/h/hospital", "/h/hospital_room",
            "/h/hot_spring", "/h/hotel/outdoor", "/h/hotel_room", "/h/house", "/h/hunting_lodge/outdoor",
            "/i/ice_cream_parlor", "/i/ice_floe", "/i/ice_shelf", "/i/ice_skating_rink/indoor", "/i/ice_skating_rink/outdoor",
            "/i/iceberg", "/i/igloo", "/i/industrial_area", "/i/inn/outdoor", "/i/islet",
            "/j/jacuzzi/indoor", "/j/jail_cell", "/j/japanese_garden", "/j/jewelry_shop", "/j/junkyard",
            "/k/kasbah", "/k/kennel/outdoor", "/k/kindergarden_classroom", "/k/kitchen", "/l/lagoon",
            "/l/lake/natural", "/l/landfill", "/l/landing_deck", "/l/laundromat", "/l/lawn",
            "/l/lecture_room", "/l/legislative_chamber", "/l/library/indoor", "/l/library/outdoor", "/l/lighthouse",
            "/l/living_room", "/l/loading_dock", "/l/lobby", "/l/lock_chamber", "/l/locker_room",
            "/m/mansion", "/m/manufactured_home", "/m/market/indoor", "/m/market/outdoor", "/m/marsh",
            "/m/martial_arts_gym", "/m/mausoleum", "/m/medina", "/m/mezzanine", "/m/moat/water",
            "/m/mosque/outdoor", "/m/motel", "/m/mountain", "/m/mountain_path", "/m/mountain_snowy",
            "/m/movie_theater/indoor", "/m/museum/indoor", "/m/museum/outdoor", "/m/music_studio", "/n/natural_history_museum",
            "/n/nursery", "/n/nursing_home", "/o/oast_house", "/o/ocean", "/o/office",
            "/o/office_building", "/o/office_cubicles", "/o/oilrig", "/o/operating_room", "/o/orchard",
            "/o/orchestra_pit", "/p/pagoda", "/p/palace", "/p/pantry", "/p/park",
            "/p/parking_garage/indoor", "/p/parking_garage/outdoor", "/p/parking_lot", "/p/pasture", "/p/patio",
            "/p/pavilion", "/p/pet_shop", "/p/pharmacy", "/p/phone_booth", "/p/physics_laboratory",
            "/p/picnic_area", "/p/pier", "/p/pizzeria", "/p/playground", "/p/playroom",
            "/p/plaza", "/p/pond", "/p/porch", "/p/promenade", "/p/pub/indoor",
            "/r/racecourse", "/r/raceway", "/r/raft", "/r/railroad_track", "/r/rainforest",
            "/r/reception", "/r/recreation_room", "/r/repair_shop", "/r/residential_neighborhood", "/r/restaurant",
            "/r/restaurant_kitchen", "/r/restaurant_patio", "/r/rice_paddy", "/r/river", "/r/rock_arch",
            "/r/roof_garden", "/r/rope_bridge", "/r/ruin", "/r/runway", "/s/sandbox",
            "/s/sauna", "/s/schoolhouse", "/s/science_museum", "/s/server_room", "/s/shed",
            "/s/shoe_shop", "/s/shopfront", "/s/shopping_mall/indoor", "/s/shower", "/s/ski_resort",
            "/s/ski_slope", "/s/sky", "/s/skyscraper", "/s/slum", "/s/snowfield",
            "/s/soccer_field", "/s/stable", "/s/stadium/baseball", "/s/stadium/football", "/s/stadium/soccer",
            "/s/stage/indoor", "/s/stage/outdoor", "/s/staircase", "/s/storage_room", "/s/street",
            "/s/subway_station/platform", "/s/supermarket", "/s/sushi_bar", "/s/swamp", "/s/swimming_hole",
            "/s/swimming_pool/indoor", "/s/swimming_pool/outdoor", "/s/synagogue/outdoor", "/t/television_room", "/t/television_studio",
            "/t/temple/asia", "/t/throne_room", "/t/ticket_booth", "/t/topiary_garden", "/t/tower",
            "/t/toyshop", "/t/train_interior", "/t/train_station/platform", "/t/tree_farm", "/t/tree_house",
            "/t/trench", "/t/tundra", "/u/underwater/ocean_deep", "/u/utility_room", "/v/valley",
            "/v/vegetable_garden", "/v/veterinarians_office", "/v/viaduct", "/v/village", "/v/vineyard",
            "/v/volcano", "/v/volleyball_court/outdoor", "/w/waiting_room", "/w/water_park", "/w/water_tower",
            "/w/waterfall", "/w/watering_hole", "/w/wave", "/w/wet_bar", "/w/wheat_field",
            "/w/wind_farm", "/w/windmill", "/y/yard", "/y/youth_hostel", "/z/zen_garden"
        };
    }
}
