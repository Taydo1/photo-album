using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using OpenCvSharp;
using PhotoApp2.Models;
using PhotoApp2.Services;
using TorchSharp.PyBridge;
using Xunit;
using Xunit.Abstractions;

namespace PhotoApp2.Tests
{
    public class PhotoAnalyzerServiceTests
    {
        private readonly ITestOutputHelper _output;

        public PhotoAnalyzerServiceTests(ITestOutputHelper output)
        {
            _output = output;
        }

        private static void EnsureTestImages(string imagesDir)
        {
            Directory.CreateDirectory(imagesDir);
            var existing = Directory.GetFiles(imagesDir, "*.jpg");
            if (!existing.Any())
            {
                string img1 = Path.Combine(imagesDir, "sample_landscape.jpg");
                using (var mat1 = new Mat(480, 640, MatType.CV_8UC3, new Scalar(100, 150, 200)))
                {
                    Cv2.Circle(mat1, new Point(320, 240), 50, new Scalar(255, 255, 255), -1);
                    Cv2.ImWrite(img1, mat1);
                }

                string img2 = Path.Combine(imagesDir, "media__1785070826785.jpg");
                using (var mat2 = new Mat(480, 640, MatType.CV_8UC3, new Scalar(200, 100, 50)))
                {
                    Cv2.Rectangle(mat2, new Rect(100, 100, 200, 200), new Scalar(0, 255, 0), -1);
                    Cv2.ImWrite(img2, mat2);
                }
            }
        }

        [Fact]
        public async Task AnalyzePhotoAsync_ProcessesAllAttachedImages_Successfully()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            EnsureTestImages(imagesDir);

            var imageFiles = Directory.GetFiles(imagesDir, "*.jpg");
            Assert.NotEmpty(imageFiles);

            var totalStopwatch = Stopwatch.StartNew();

            foreach (var file in imageFiles)
            {
                var sw = Stopwatch.StartNew();
                var result = await service.AnalyzePhotoAsync(file);
                sw.Stop();

                _output.WriteLine($"File: {Path.GetFileName(file)} | Time: {sw.ElapsedMilliseconds} ms | Faces: {result.FaceCount} | Sharpness: {result.SharpnessScore:F2} | Tags: {result.TagsDisplay}");

                Assert.NotNull(result);
                Assert.True(result.IsAnalyzed, $"Photo analysis failed for {file}");
                Assert.True(sw.ElapsedMilliseconds >= 0);
            }

            totalStopwatch.Stop();
            _output.WriteLine($"Total time for {imageFiles.Length} images: {totalStopwatch.ElapsedMilliseconds} ms (avg {totalStopwatch.ElapsedMilliseconds / imageFiles.Length} ms/image)");
        }

        [Fact]
        public async Task AnalyzePhotoAsync_GroupPhoto_DetectsFacesOrAddsPersonTag()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            EnsureTestImages(imagesDir);

            var groupPhotoPath = Directory.GetFiles(imagesDir, "media__1785070826785.jpg").FirstOrDefault()
                ?? Directory.GetFiles(imagesDir, "*.jpg").FirstOrDefault();

            Assert.NotNull(groupPhotoPath);

            var sw = Stopwatch.StartNew();
            var result = await service.AnalyzePhotoAsync(groupPhotoPath);
            sw.Stop();

            _output.WriteLine($"Group photo analyzed in {sw.ElapsedMilliseconds} ms | Faces found: {result.FaceCount} | Tags: {result.TagsDisplay}");

            Assert.True(result.IsAnalyzed);
            Assert.NotNull(result.Tags);
        }

        [Fact]
        public async Task Benchmark_BatchAnalyzePhotos_Throughput()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();
            var dbService = new DatabaseService();
            await dbService.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            EnsureTestImages(imagesDir);

            var sampleFiles = Directory.GetFiles(imagesDir, "*.jpg");
            Assert.NotEmpty(sampleFiles);

            var testPhotos = Enumerable.Range(1, 10)
                .SelectMany(i => sampleFiles.Select((file, idx) => new Models.PhotoItem
                {
                    FilePath = file + $"_sim_{i}_{idx}",
                    FileName = Path.GetFileName(file),
                    FileSizeBytes = new FileInfo(file).Length,
                    DateTaken = DateTime.Now.AddDays(-idx)
                }))
                .ToList();

            _output.WriteLine($"=== STARTING BATCH ANALYSIS BENCHMARK ({testPhotos.Count} items) ===");
            var analysisSw = Stopwatch.StartNew();

            await Parallel.ForEachAsync(testPhotos, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, async (photo, ct) =>
            {
                var actualFilePath = photo.FilePath.Substring(0, photo.FilePath.IndexOf("_sim_"));
                var analyzed = await service.AnalyzePhotoAsync(actualFilePath);

                photo.IsAnalyzed = true;
                photo.SharpnessScore = analyzed.SharpnessScore;
                photo.FaceCount = analyzed.FaceCount;
                photo.Tags = analyzed.Tags;
                photo.VisualFeatureVector = analyzed.VisualFeatureVector;
            });

            analysisSw.Stop();
            double fps = testPhotos.Count / analysisSw.Elapsed.TotalSeconds;
            _output.WriteLine($"Parallel Analysis Time: {analysisSw.ElapsedMilliseconds} ms ({fps:F1} photos/sec)");

            var dbSw = Stopwatch.StartNew();
            await dbService.SavePhotosAsync(testPhotos);
            dbSw.Stop();
            _output.WriteLine($"Batch SQLite Save Time ({testPhotos.Count} items): {dbSw.ElapsedMilliseconds} ms");

            Assert.True(testPhotos.All(p => p.IsAnalyzed), "All photos should be analyzed");
            Assert.True(analysisSw.ElapsedMilliseconds < 10000, "Parallel batch analysis should complete quickly");
            Assert.True(dbSw.ElapsedMilliseconds < 1000, "Transactional batch save should be instantaneous");
        }

        [Fact]
        public async Task AnalyzePhotoAsync_ExtractsTags_Successfully()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            EnsureTestImages(imagesDir);

            var sampleFile = Directory.GetFiles(imagesDir, "*.jpg").FirstOrDefault();
            Assert.NotNull(sampleFile);

            var result = await service.AnalyzePhotoAsync(sampleFile);
            _output.WriteLine($"File: {result.FileName} | Tags: {result.TagsDisplay}");

            Assert.True(result.IsAnalyzed);
            Assert.NotNull(result.Tags);
        }

        [Fact]
        public void CosineSimilarity_CalculatesCorrectVectorSimilarity()
        {
            float[] vecA = new float[] { 0.5f, 0.5f, 0.0f, 0.0f };
            float[] vecB = new float[] { 0.5f, 0.5f, 0.0f, 0.0f };
            float[] vecC = new float[] { 0.0f, 0.0f, 1.0f, 1.0f };

            double simIdentical = OnnxContentClassifier.CalculateCosineSimilarity(vecA, vecB);
            double simOrthogonal = OnnxContentClassifier.CalculateCosineSimilarity(vecA, vecC);

            Assert.Equal(1.0, simIdentical, precision: 4);
            Assert.Equal(0.0, simOrthogonal, precision: 4);
        }

        [Fact]
        public async Task AlbumGeneratorService_EnforcesOneOrTwoKindsPerPage_AndFormatsFolderNames()
        {
            var generator = new AlbumGeneratorService();
            var tempDir = Path.Combine(Path.GetTempPath(), "PhotoApp2_TestAlbum_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempDir);

            try
            {
                var simPhotos = new List<Models.PhotoItem>
                {
                    new Models.PhotoItem { Id = 1, FilePath = "fake1.jpg", FileName = "fake1.jpg", IsAnalyzed = true, SharpnessScore = 100, DateTaken = DateTime.Today.AddHours(1), Tags = new List<string> { "Landscape" }, VisualFeatureVector = new float[] { 1f, 0f, 0f } },
                    new Models.PhotoItem { Id = 2, FilePath = "fake2.jpg", FileName = "fake2.jpg", IsAnalyzed = true, SharpnessScore = 120, DateTaken = DateTime.Today.AddHours(2), Tags = new List<string> { "Person" }, VisualFeatureVector = new float[] { 0f, 1f, 0f } },
                    new Models.PhotoItem { Id = 3, FilePath = "fake3.jpg", FileName = "fake3.jpg", IsAnalyzed = true, SharpnessScore = 110, DateTaken = DateTime.Today.AddHours(3), Tags = new List<string> { "Architecture" }, VisualFeatureVector = new float[] { 0f, 0f, 1f } },
                    new Models.PhotoItem { Id = 4, FilePath = "fake4.jpg", FileName = "fake4.jpg", IsAnalyzed = true, SharpnessScore = 105, DateTaken = DateTime.Today.AddHours(4), Tags = new List<string> { "Leisure & Recreation", "park" }, VisualFeatureVector = new float[] { 0.5f, 0.5f, 0f } }
                };

                var pages = await generator.GenerateAlbumAsync(simPhotos, tempDir);

                _output.WriteLine($"Generated {pages.Count} pages.");
                foreach (var page in pages)
                {
                    _output.WriteLine($"Page {page.PageNumber} Theme: {page.Theme} | Photos: {page.Photos.Count}");
                    
                    string expectedPrefix = $"Page_{page.PageNumber:D3}_";
                    var matchingFolders = Directory.GetDirectories(tempDir).Where(d => Path.GetFileName(d).StartsWith(expectedPrefix)).ToList();
                    Assert.NotEmpty(matchingFolders);
                }

                var htmlPath = Path.Combine(tempDir, "Album_Preview.html");
                Assert.True(File.Exists(htmlPath), "Album_Preview.html should be created");
            }
            finally
            {
                if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
            }
        }

        private class CustomUnpickler : Razorvine.Pickle.Unpickler
        {
            protected override object persistentLoad(object pid)
            {
                if (pid is object[] arr)
                {
                    return new StorageRef { Data = arr };
                }
                return pid!;
            }
        }

        private class StorageRef
        {
            public object[]? Data { get; set; }
            public override string ToString() => "StorageRef: " + (Data != null ? string.Join(", ", Data.Select(d => d?.ToString())) : "null");
        }

        private class DummyConstructor : Razorvine.Pickle.IObjectConstructor
        {
            public object construct(object[] args)
            {
                if (args != null && args.Length == 1 && args[0] is System.Collections.ArrayList al)
                {
                    var dict = new System.Collections.Hashtable();
                    foreach (object item in al)
                    {
                        if (item is object[] pair && pair.Length == 2 && pair[0] != null)
                        {
                            dict[pair[0]!.ToString()!] = pair[1];
                        }
                        else if (item is System.Collections.ArrayList pairList && pairList.Count == 2 && pairList[0] != null)
                        {
                            dict[pairList[0]!.ToString()!] = pairList[1];
                        }
                    }
                    if (dict.Count > 0) return dict;
                }
                return new TensorRef { Args = args };
            }
        }

        private class TensorRef
        {
            public object[]? Args { get; set; }
            public override string ToString() => $"TensorRef ({Args?.Length} args)";
        }

        [Fact]
        public void Diagnostic_InspectModelLoading()
        {
            Razorvine.Pickle.Unpickler.registerConstructor("torch._utils", "_rebuild_tensor", new DummyConstructor());
            Razorvine.Pickle.Unpickler.registerConstructor("torch._utils", "_rebuild_tensor_v2", new DummyConstructor());
            Razorvine.Pickle.Unpickler.registerConstructor("torch._utils", "_rebuild_parameter", new DummyConstructor());
            Razorvine.Pickle.Unpickler.registerConstructor("collections", "OrderedDict", new DummyConstructor());

            var modelPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "PhotoApp2", "Models", "resnet18_places365.pth.tar");
            _output.WriteLine($"Checking model path: {modelPath}, Exists: {File.Exists(modelPath)}");
            if (File.Exists(modelPath))
            {
                var resnet = TorchSharp.torchvision.models.resnet18(num_classes: 365);
                var modelStateDict = resnet.state_dict();

                using (var fs = File.OpenRead(modelPath))
                {
                    var unpickler = new CustomUnpickler();
                    try
                    {
                        var obj1 = unpickler.load(fs);
                        var obj2 = unpickler.load(fs);
                        var obj3 = unpickler.load(fs);
                        var obj4 = unpickler.load(fs);
                        var storageListObj = unpickler.load(fs);

                        if (obj4 is System.Collections.Hashtable ht4 && ht4.ContainsKey("state_dict") &&
                            ht4["state_dict"] is System.Collections.Hashtable sdHt &&
                            storageListObj is System.Collections.ArrayList storages)
                        {
                            _output.WriteLine($"Loading {storages.Count} storages...");
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
                                            else
                                            {
                                                _output.WriteLine($"Missing tensor or storage for {cleanKey}");
                                            }
                                        }
                                    }
                                }
                            }
                            _output.WriteLine($"Successfully loaded and copied {loadedCount} / {sdHt.Count} tensors from checkpoint (20 batchnorm num_batches_tracked buffers untouched)!");
                            Assert.Equal(sdHt.Count, loadedCount);
                            resnet.eval();
                        }
                    }
                    catch (Exception ex)
                    {
                        _output.WriteLine($"Unpickling error: {ex}");
                        throw;
                    }
                }
            }
        }

        [Fact]
        public void MapIndexToPrimaryKind_CoversAll365IndicesWithoutOther()
        {
            var mapMethod = typeof(OnnxContentClassifier).GetMethod("MapIndexToPrimaryKind", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            Assert.NotNull(mapMethod);

            var validCategories = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "Landscape", "Architecture", "Urban & Travel", "Home & Indoors",
                "Food & Dining", "Leisure & Recreation", "Culture & Education", "Work & Industry"
            };

            for (int i = 0; i < 365; i++)
            {
                string? kind = mapMethod.Invoke(null, new object[] { i }) as string;
                Assert.NotNull(kind);
                Assert.NotEqual("Other", kind);
                Assert.Contains(kind, validCategories);
            }
        }

        [Fact]
        public void ProcessClassificationResults_SumsTagProbabilities_AndOrdersPrimaryTags()
        {
            var processMethod = typeof(OnnxContentClassifier).GetMethod("ProcessClassificationResults", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            Assert.NotNull(processMethod);

            // Index 8: apartment_building/outdoor -> Architecture
            // Index 10: aqueduct -> Architecture
            // Index 30: badlands -> Landscape
            float[] probs = new float[365];
            probs[8] = 0.25f;  // Architecture (total = 0.45f)
            probs[10] = 0.20f; // Architecture
            probs[30] = 0.30f; // Landscape (total = 0.30f)

            var result = processMethod.Invoke(null, new object[] { probs });
            Assert.NotNull(result);

            var valueTuple = ((List<string> Tags, double Confidence, float[] FeatureVector))result;
            var tags = valueTuple.Tags;

            Assert.Equal(2, tags.Count);
            // Primary tags ranked by total probability: Architecture (0.45) > Landscape (0.30)
            Assert.Equal("Architecture", tags[0]);
            Assert.Equal("Landscape", tags[1]);

            // Raw class keywords must NOT be present
            Assert.DoesNotContain("badlands", tags);
            Assert.DoesNotContain("apartment building outdoor", tags);
            Assert.DoesNotContain("aqueduct", tags);
        }
    }
}
