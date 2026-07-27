using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace PhotoApp2.Services
{
    public class SiglipModelData
    {
        [JsonPropertyName("LogitScale")]
        public float LogitScale { get; set; } = 4.765f;

        [JsonPropertyName("LogitBias")]
        public float LogitBias { get; set; } = -12.932f;

        [JsonPropertyName("EmbeddingDimension")]
        public int EmbeddingDimension { get; set; } = 768;

        [JsonPropertyName("Categories")]
        public List<SiglipCategoryItem> Categories { get; set; } = new List<SiglipCategoryItem>();
    }

    public class SiglipCategoryItem
    {
        [JsonPropertyName("Prompt")]
        public string Prompt { get; set; } = string.Empty;

        [JsonPropertyName("Tags")]
        public List<string> Tags { get; set; } = new List<string>();

        [JsonPropertyName("Embedding")]
        public float[] Embedding { get; set; } = Array.Empty<float>();
    }

    public class OnnxContentClassifier : IDisposable
    {
        private InferenceSession? _session;
        private SiglipModelData? _modelData;
        private readonly object _inferLock = new object();
        private bool _isGpuEnabled;
        private int _inputWidth = 224;
        private int _inputHeight = 224;
        private string _inputName = "pixel_values";

        private static List<SiglipCategoryItem>? _cachedCategories;
        private static readonly object _cacheLock = new object();

        public bool IsGpuEnabled => _isGpuEnabled;
        public bool IsInitialized => _session != null && _modelData != null;

        internal static List<SiglipCategoryItem> GetOrLoadCategories()
        {
            if (_cachedCategories != null) return _cachedCategories;
            lock (_cacheLock)
            {
                if (_cachedCategories != null) return _cachedCategories;
                try
                {
                    var baseDir = AppDomain.CurrentDomain.BaseDirectory;
                    string jsonPath = Path.Combine(baseDir, "Assets", "Models", "siglip_categories.json");
                    if (!File.Exists(jsonPath))
                    {
                        // Fallback resolution for test runners or alternative working directories
                        string altPath = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "PhotoApp2", "Assets", "Models", "siglip_categories.json"));
                        if (File.Exists(altPath)) jsonPath = altPath;
                        else
                        {
                            var appDataModelPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "PhotoApp2", "Models", "siglip_categories.json");
                            if (File.Exists(appDataModelPath)) jsonPath = appDataModelPath;
                        }
                    }

                    if (File.Exists(jsonPath))
                    {
                        string jsonContent = File.ReadAllText(jsonPath);
                        var modelData = JsonSerializer.Deserialize<SiglipModelData>(jsonContent);
                        if (modelData != null && modelData.Categories != null)
                        {
                            _cachedCategories = modelData.Categories;
                            return _cachedCategories;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error loading categories from json: {ex.Message}");
                }
                _cachedCategories = new List<SiglipCategoryItem>();
                return _cachedCategories;
            }
        }

        public async Task InitializeAsync(string? modelPath = null)
        {
            if (_session != null && _modelData != null) return;

            await Task.Run(async () =>
            {
                try
                {
                    var baseDir = AppDomain.CurrentDomain.BaseDirectory;
                    var appDataDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "PhotoApp2", "Models");
                    Directory.CreateDirectory(appDataDir);

                    // 1. Load precomputed SigLIP text category embeddings from our authoritative asset
                    string jsonPath = Path.Combine(baseDir, "Assets", "Models", "siglip_categories.json");
                    if (!File.Exists(jsonPath))
                    {
                        string altPath = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "PhotoApp2", "Assets", "Models", "siglip_categories.json"));
                        if (File.Exists(altPath)) jsonPath = altPath;
                        else if (File.Exists(Path.Combine(appDataDir, "siglip_categories.json"))) jsonPath = Path.Combine(appDataDir, "siglip_categories.json");
                    }

                    if (File.Exists(jsonPath))
                    {
                        string jsonContent = await File.ReadAllTextAsync(jsonPath);
                        _modelData = JsonSerializer.Deserialize<SiglipModelData>(jsonContent);
                        if (_modelData != null && _modelData.Categories != null)
                        {
                            lock (_cacheLock)
                            {
                                _cachedCategories = _modelData.Categories;
                            }
                        }
                        Debug.WriteLine($"Loaded {_modelData?.Categories?.Count} SigLIP category embeddings from {jsonPath}");
                    }
                    else
                    {
                        Debug.WriteLine($"WARNING: SigLIP categories file not found at {jsonPath}. Initializing built-in fallback categories.");
                        _modelData = CreateFallbackModelData();
                    }

                    // 2. Locate or download SigLIP vision ONNX model
                    string? rawModelPath = modelPath;
                    if (string.IsNullOrEmpty(rawModelPath))
                    {
                        rawModelPath = Path.Combine(baseDir, "Assets", "Models", "siglip-base-patch16-224-vision.onnx");
                        if (!File.Exists(rawModelPath))
                        {
                            rawModelPath = Path.Combine(baseDir, "Assets", "Models", "vision_model.onnx");
                        }
                    }

                    if (!File.Exists(rawModelPath))
                    {
                        var downloadPath = Path.Combine(appDataDir, "siglip-base-patch16-224-vision.onnx");
                        if (!File.Exists(downloadPath))
                        {
                            Debug.WriteLine("Downloading SigLIP Base ONNX vision model from Hugging Face...");
                            try
                            {
                                using var client = new HttpClient();
                                client.Timeout = TimeSpan.FromMinutes(10);
                                var data = await client.GetByteArrayAsync("https://huggingface.co/Xenova/siglip-base-patch16-224/resolve/main/onnx/vision_model.onnx");
                                await File.WriteAllBytesAsync(downloadPath, data);
                                Debug.WriteLine($"Downloaded SigLIP Base vision model ({data.Length / 1024 / 1024} MB)");
                            }
                            catch (Exception downloadEx)
                            {
                                Debug.WriteLine($"Auto-download of SigLIP vision model failed: {downloadEx.Message}");
                            }
                        }
                        if (File.Exists(downloadPath))
                            rawModelPath = downloadPath;
                    }

                    if (!File.Exists(rawModelPath))
                    {
                        Debug.WriteLine("WARNING: SigLIP vision ONNX model file unavailable. Scene classification will return defaults.");
                        return;
                    }

                    // 3. Initialize ONNX Runtime InferenceSession with DirectML hardware GPU acceleration
                    using var sessionOptions = new SessionOptions();
                    try
                    {
                        sessionOptions.AppendExecutionProvider_DML(0);
                        _isGpuEnabled = true;
                        Debug.WriteLine("ONNX Runtime initialized with DirectML GPU acceleration for SigLIP.");
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"DirectML unavailable, falling back to CPU threads: {ex.Message}");
                        _isGpuEnabled = false;
                    }

                    _session = new InferenceSession(rawModelPath, sessionOptions);

                    if (_session.InputMetadata.Count > 0)
                    {
                        _inputName = _session.InputMetadata.Keys.First();
                        var shape = _session.InputMetadata[_inputName].Dimensions;
                        if (shape.Length == 4 && shape[2] > 0 && shape[3] > 0)
                        {
                            _inputHeight = shape[2];
                            _inputWidth = shape[3];
                        }
                    }

                    Debug.WriteLine($"SigLIP vision model loaded successfully. Expected input: '{_inputName}' [{_inputWidth}x{_inputHeight}].");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error initializing OnnxContentClassifier: {ex.Message}");
                    Console.WriteLine($"Error initializing OnnxContentClassifier: {ex.Message}");
                }
            });
        }

        public Task<(List<string> Tags, double Confidence, float[] FeatureVector)> ClassifySceneAsync(Mat inputMat)
            => Task.Run(() => ClassifyScene(inputMat));

        public (List<string> Tags, double Confidence, float[] FeatureVector) ClassifyScene(Mat inputMat)
        {
            if (_session == null || _modelData == null || _modelData.Categories.Count == 0 || inputMat.Empty())
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

                // Prepare SigLIP normalized input float array [1, 3, 224, 224]
                // SigLIP uses Mean (0.5, 0.5, 0.5) and Std (0.5, 0.5, 0.5) -> (val / 127.5f) - 1.0f
                var inputTensor = new DenseTensor<float>(new[] { 1, 3, _inputHeight, _inputWidth });

                unsafe
                {
                    byte* ptr = (byte*)rgb.Data;
                    int stride = (int)rgb.Step();

                    for (int y = 0; y < _inputHeight; y++)
                    {
                        byte* row = ptr + (y * stride);
                        for (int x = 0; x < _inputWidth; x++)
                        {
                            int idx = x * 3;
                            inputTensor[0, 0, y, x] = ((row[idx] / 127.5f) - 1.0f);     // R
                            inputTensor[0, 1, y, x] = ((row[idx + 1] / 127.5f) - 1.0f); // G
                            inputTensor[0, 2, y, x] = ((row[idx + 2] / 127.5f) - 1.0f); // B
                        }
                    }
                }

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
                };

                float[] rawVisionVec;
                lock (_inferLock)
                {
                    using var results = _session.Run(inputs);
                    NamedOnnxValue? targetOutput = null;

                    foreach (var res in results)
                    {
                        if (res.Name == "pooler_output" || res.Name == "image_embeds" || res.Name == "embedding" || res.Name == "output")
                        {
                            targetOutput = res;
                            break;
                        }
                    }
                    if (targetOutput == null)
                    {
                        foreach (var res in results)
                        {
                            var t = res.AsTensor<float>();
                            if (t.Length == _modelData.EmbeddingDimension)
                            {
                                targetOutput = res;
                                break;
                            }
                        }
                    }
                    if (targetOutput == null)
                        targetOutput = results.First();

                    rawVisionVec = targetOutput.AsTensor<float>().ToArray();
                }

                float[] visionEmbed = NormalizeVector(rawVisionVec);
                float[] probs = CalculateSigmoidProbabilities(visionEmbed, _modelData);

                return ProcessClassificationResults(probs, visionEmbed);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error during SigLIP ONNX scene classification: {ex.Message}");
                return (new List<string> { "Other" }, 0.0, Array.Empty<float>());
            }
        }

        private static float[] NormalizeVector(float[] vector)
        {
            if (vector == null || vector.Length == 0) return Array.Empty<float>();
            double norm = 0.0;
            for (int i = 0; i < vector.Length; i++)
                norm += vector[i] * vector[i];
            norm = Math.Sqrt(norm);
            if (norm <= 0.0) return vector;

            float[] normalized = new float[vector.Length];
            for (int i = 0; i < vector.Length; i++)
                normalized[i] = (float)(vector[i] / norm);
            return normalized;
        }

        private static float[] CalculateSigmoidProbabilities(float[] visionEmbed, SiglipModelData modelData)
        {
            int count = modelData.Categories.Count;
            float[] probs = new float[count];
            double expScale = Math.Exp(modelData.LogitScale);
            double bias = modelData.LogitBias;

            for (int i = 0; i < count; i++)
            {
                float[] textEmbed = modelData.Categories[i].Embedding;
                double dot = 0.0;
                int len = Math.Min(visionEmbed.Length, textEmbed.Length);
                for (int j = 0; j < len; j++)
                {
                    dot += visionEmbed[j] * textEmbed[j];
                }

                double logit = (dot * expScale) + bias;
                double prob = 1.0 / (1.0 + Math.Exp(-logit));
                probs[i] = (float)prob;
            }

            return probs;
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

        private static (List<string> Tags, double Confidence, float[] FeatureVector) ProcessClassificationResults(float[] probs, float[]? featureVector = null)
        {
            if (probs == null || probs.Length == 0)
                return (new List<string> { "Other" }, 0.0, featureVector ?? Array.Empty<float>());

            var categories = GetOrLoadCategories();
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

            // Because SigLIP evaluates each category independently via Sigmoid loss, aggregate all tags from categories scoring above threshold
            var tagScores = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < probs.Length && i < categories.Count; i++)
            {
                var categoryTags = categories[i].Tags;
                if (categoryTags == null) continue;

                foreach (string tag in categoryTags)
                {
                    if (!string.IsNullOrEmpty(tag) && tag != "Other")
                    {
                        if (tagScores.TryGetValue(tag, out float currentProb))
                        {
                            if (probs[i] > currentProb)
                            {
                                tagScores[tag] = probs[i];
                            }
                        }
                        else
                        {
                            tagScores[tag] = probs[i];
                        }
                    }
                }
            }

            var tagsList = new List<string>();

            // Filter tags with aggregated probability >= 0.20f, ordered descending by probability
            var validTagPairs = tagScores
                .Where(kvp => kvp.Value >= 0.20f)
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
                // Fallback to top index's tags if no category reached threshold
                var topTags = MapIndexToTags(topIndex);
                if (topTags != null)
                {
                    foreach (var t in topTags)
                    {
                        if (!string.IsNullOrEmpty(t) && t != "Other" && !tagsList.Contains(t, StringComparer.OrdinalIgnoreCase))
                            tagsList.Add(t);
                    }
                }
            }

            if (!tagsList.Any())
            {
                tagsList.Add("Other");
            }

            return (tagsList, confidence, featureVector ?? probs);
        }

        private static List<string> MapIndexToTags(int index)
        {
            var categories = GetOrLoadCategories();
            if (index >= 0 && index < categories.Count && categories[index].Tags != null)
            {
                return categories[index].Tags;
            }
            return new List<string> { "Other" };
        }

        private static SiglipModelData CreateFallbackModelData()
        {
            var data = new SiglipModelData();
            data.Categories.Add(new SiglipCategoryItem
            {
                Prompt = "fallback",
                Tags = new List<string> { "Other" },
                Embedding = new float[768]
            });
            return data;
        }

        public void Dispose()
        {
            _session?.Dispose();
            _session = null;
            GC.SuppressFinalize(this);
        }
    }
}
