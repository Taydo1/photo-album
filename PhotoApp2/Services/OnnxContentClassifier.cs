using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace PhotoApp2.Services
{
    public class OnnxContentClassifier : IDisposable
    {
        private InferenceSession? _session;
        private readonly object _inferLock = new object();
        private bool _isGpuEnabled;
        private int _inputWidth = 224;
        private int _inputHeight = 224;
        private string _inputName = "input";

        public bool IsGpuEnabled => _isGpuEnabled;
        public bool IsInitialized => _session != null;

        public async Task InitializeAsync(string? modelPath = null)
        {
            if (_session != null) return;

            await Task.Run(async () =>
            {
                try
                {
                    if (string.IsNullOrEmpty(modelPath))
                    {
                        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
                        modelPath = Path.Combine(baseDir, "Assets", "Models", "mobilenetv2-12.onnx");
                    }

                    // Fallback to local app data or auto-download if missing from base directory
                    if (!File.Exists(modelPath))
                    {
                        var appDataDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "PhotoApp2", "Models");
                        Directory.CreateDirectory(appDataDir);
                        var localCopy = Path.Combine(appDataDir, "mobilenetv2-12.onnx");
                        if (!File.Exists(localCopy))
                        {
                            Debug.WriteLine("Downloading mobilenetv2-12.onnx from ONNX Model Zoo...");
                            using var client = new HttpClient();
                            var data = await client.GetByteArrayAsync("https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx");
                            await File.WriteAllBytesAsync(localCopy, data);
                        }
                        modelPath = localCopy;
                    }

                    if (!File.Exists(modelPath))
                    {
                        Debug.WriteLine($"ONNX Scene Model not found at: {modelPath}");
                        return;
                    }

                    using var sessionOptions = new SessionOptions();
                    try
                    {
                        sessionOptions.AppendExecutionProvider_DML(0);
                        _isGpuEnabled = true;
                        Debug.WriteLine("OnnxContentClassifier initialized with DirectML GPU acceleration.");
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"DirectML unavailable for Content Classifier, falling back to CPU: {ex.Message}");
                        _isGpuEnabled = false;
                    }

                    _session = new InferenceSession(modelPath, sessionOptions);
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
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error initializing OnnxContentClassifier: {ex.Message}");
                }
            });
        }

        public Task<(string Keywords, string PrimaryKind, double Confidence)> ClassifySceneAsync(Mat inputMat)
            => Task.Run(() => ClassifyScene(inputMat));

        public (string Keywords, string PrimaryKind, double Confidence) ClassifyScene(Mat inputMat)
        {
            if (_session == null || inputMat.Empty())
                return ("", "Other", 0.0);

            try
            {
                using var resized = new Mat();
                Cv2.Resize(inputMat, resized, new Size(_inputWidth, _inputHeight), 0, 0, InterpolationFlags.Linear);

                var tensor = new DenseTensor<float>(new[] { 1, 3, _inputHeight, _inputWidth });

                // ImageNet standard normalization: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                const float meanR = 0.485f * 255.0f;
                const float meanG = 0.456f * 255.0f;
                const float meanB = 0.406f * 255.0f;
                const float invStdR = 1.0f / (0.229f * 255.0f);
                const float invStdG = 1.0f / (0.224f * 255.0f);
                const float invStdB = 1.0f / (0.225f * 255.0f);

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

                int stride = (int)rgb.Step();
                int bufferSize = _inputHeight * stride;
                byte[] buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(bufferSize);
                try
                {
                    System.Runtime.InteropServices.Marshal.Copy(rgb.Data, buffer, 0, bufferSize);
                    for (int y = 0; y < _inputHeight; y++)
                    {
                        int rowOffset = y * stride;
                        for (int x = 0; x < _inputWidth; x++)
                        {
                            int idx = rowOffset + (x * 3);
                            tensor[0, 0, y, x] = (buffer[idx] - meanR) * invStdR;
                            tensor[0, 1, y, x] = (buffer[idx + 1] - meanG) * invStdG;
                            tensor[0, 2, y, x] = (buffer[idx + 2] - meanB) * invStdB;
                        }
                    }
                }
                finally
                {
                    System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
                }

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(_inputName, tensor)
                };

                int topClassIdx = -1;
                float maxVal = float.NegativeInfinity;

                lock (_inferLock)
                {
                    using var results = _session.Run(inputs);
                    var outputVal = results.First().AsTensor<float>();
                    int numClasses = outputVal.Dimensions[1];

                    for (int i = 0; i < numClasses; i++)
                    {
                        float val = outputVal[0, i];
                        if (val > maxVal)
                        {
                            maxVal = val;
                            topClassIdx = i;
                        }
                    }
                }

                return MapIndexToCategory(topClassIdx, maxVal);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error during GPU OnnxContentClassifier inference: {ex.Message}");
                return ("", "Other", 0.0);
            }
        }

        private static (string Keywords, string PrimaryKind, double Confidence) MapIndexToCategory(int index, float logitScore)
        {
            // Approximate confidence via logit scaling
            double confidence = Math.Min(1.0, Math.Max(0.0, (logitScore + 5.0) / 15.0));

            // ImageNet standard scene mapping for photo albums
            if (index >= 970 && index <= 985)
            {
                // Outdoor landscapes & nature scenes (alp, cliff, seashore, lakeside, coral reef, valley, volcano, rapeseed, daisy)
                if (index == 973 || index == 975 || index == 976 || index == 977 || index == 978)
                    return ("waterfront, beautiful landscape, beach", "Landscape", confidence);
                if (index == 970 || index == 972 || index == 979 || index == 980)
                    return ("mountain, beautiful landscape, nature", "Landscape", confidence);
                return ("nature, beautiful landscape", "Landscape", confidence);
            }

            // Architecture, Landmarks, Urban & Vehicles
            if (index == 483 || index == 488 || index == 497 || index == 510 || index == 536 || 
                index == 538 || index == 609 || index == 668 || index == 700 || index == 706 || 
                index == 738 || index == 780 || index == 839 || index == 868 || index == 878 || index == 907)
            {
                return ("architecture, landmark", "Architecture", confidence);
            }
            if (index == 404 || index == 436 || index == 449 || index == 468 || index == 511 || 
                index == 656 || index == 717 || index == 817 || index == 841 || index == 867)
            {
                return ("travel, transport, city street", "Architecture & City", confidence);
            }

            // Animals & Wildlife (0 - 399)
            if (index >= 0 && index <= 399)
            {
                if (index >= 151 && index <= 293)
                    return ("animal, pet", "Animal", confidence);
                return ("animal, wildlife", "Animal", confidence);
            }

            // Food & Dining & Celebration (924 - 969, plus restaurant 762)
            if ((index >= 924 && index <= 969) || index == 762)
            {
                return ("dining, food & drinks, celebration", "Event", confidence);
            }

            // General fallbacks
            return ("outdoor scene", "Other", confidence);
        }

        public void Dispose()
        {
            _session?.Dispose();
            _session = null;
            GC.SuppressFinalize(this);
        }
    }
}
