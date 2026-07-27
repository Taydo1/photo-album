using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace PhotoApp2.Services
{
    public class OnnxFaceDetector : IDisposable
    {
        private InferenceSession? _session;
        private readonly object _inferLock = new object();
        private bool _isGpuEnabled;
        private int _inputWidth = 320;
        private int _inputHeight = 240;
        private string _inputName = "input";

        public bool IsGpuEnabled => _isGpuEnabled;
        public bool IsInitialized => _session != null;

        public async Task InitializeAsync(string? modelPath = null)
        {
            if (_session != null) return;

            await Task.Run(() =>
            {
                try
                {
                    if (string.IsNullOrEmpty(modelPath))
                    {
                        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
                        modelPath = Path.Combine(baseDir, "Assets", "Models", "version-RFB-320.onnx");
                    }

                    if (!File.Exists(modelPath))
                    {
                        Debug.WriteLine($"ONNX Model not found at: {modelPath}");
                        return;
                    }

                    using var sessionOptions = new SessionOptions();
                    try
                    {
                        // Enable hardware GPU acceleration via DirectX 12 DirectML (Targeting primary adapter 0)
                        sessionOptions.AppendExecutionProvider_DML(0);
                        _isGpuEnabled = true;
                        Debug.WriteLine("ONNX Runtime initialized with DirectML GPU acceleration.");
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"DirectML unavailable, falling back to high-speed CPU threads: {ex.Message}");
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

                    // Warm up DirectML GPU graph compilation with a single dummy run
                    try
                    {
                        var dummyTensor = new DenseTensor<float>(new[] { 1, 3, _inputHeight, _inputWidth });
                        var dummyInputs = new List<NamedOnnxValue>
                        {
                            NamedOnnxValue.CreateFromTensor(_inputName, dummyTensor)
                        };
                        lock (_inferLock)
                        {
                            using var dummyResults = _session.Run(dummyInputs);
                        }
                        Debug.WriteLine("DirectML graph optimization and warm-up complete.");
                    }
                    catch (Exception ex)
                    {
                        Debug.WriteLine($"DirectML warm-up error: {ex.Message}");
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error initializing OnnxFaceDetector: {ex.Message}");
                }
            });
        }

        public Task<int> DetectFacesAsync(Mat inputMat) => Task.Run(() => DetectFaces(inputMat));

        public int DetectFaces(Mat inputMat)
        {
            if (_session == null || inputMat.Empty()) return 0;

            try
            {
                using var resized = new Mat();
                // Nearest neighbor resize minimizes CPU vector clock cycles during downscaling
                Cv2.Resize(inputMat, resized, new Size(_inputWidth, _inputHeight), 0, 0, InterpolationFlags.Nearest);

                var tensor = new DenseTensor<float>(new[] { 1, 3, _inputHeight, _inputWidth });

                if (resized.Channels() == 1)
                {
                    // Grayscale fast-path: zero color conversion overhead, 1/3rd memory copying, 1/3rd arithmetic operations!
                    int stride = (int)resized.Step();
                    int bufferSize = _inputHeight * stride;
                    byte[] buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(bufferSize);
                    try
                    {
                        System.Runtime.InteropServices.Marshal.Copy(resized.Data, buffer, 0, bufferSize);
                        for (int y = 0; y < _inputHeight; y++)
                        {
                            int rowOffset = y * stride;
                            for (int x = 0; x < _inputWidth; x++)
                            {
                                float norm = (buffer[rowOffset + x] - 127.0f) / 128.0f;
                                tensor[0, 0, y, x] = norm; // R
                                tensor[0, 1, y, x] = norm; // G
                                tensor[0, 2, y, x] = norm; // B
                            }
                        }
                    }
                    finally
                    {
                        System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
                    }
                }
                else
                {
                    using var rgb = new Mat();
                    if (resized.Channels() == 3)
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
                                tensor[0, 0, y, x] = (buffer[idx] - 127.0f) / 128.0f;     // R
                                tensor[0, 1, y, x] = (buffer[idx + 1] - 127.0f) / 128.0f; // G
                                tensor[0, 2, y, x] = (buffer[idx + 2] - 127.0f) / 128.0f; // B
                            }
                        }
                    }
                    finally
                    {
                        System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
                    }
                }

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(_inputName, tensor)
                };

                int numAnchors = 0;
                var candidateRects = new List<Rect2d>();
                var candidateScores = new List<float>();
                const float confidenceThreshold = 0.65f;

                lock (_inferLock)
                {
                    using var results = _session.Run(inputs);

                    // Parse output tensors: Boxes [1, N, 4] and Scores [1, N, 2]
                    var boxesVal = results.FirstOrDefault(v => v.Name == "boxes") ?? results.ElementAt(0);
                    var scoresVal = results.FirstOrDefault(v => v.Name == "scores") ?? results.ElementAt(1);

                    var boxesTensor = boxesVal.AsTensor<float>();
                    var scoresTensor = scoresVal.AsTensor<float>();

                    numAnchors = scoresTensor.Dimensions[1];

                    for (int i = 0; i < numAnchors; i++)
                    {
                        float faceScore = scoresTensor[0, i, 1];
                        if (faceScore > confidenceThreshold)
                        {
                            // Bounding box coords normalized [xMin, yMin, xMax, yMax]
                            float x1 = boxesTensor[0, i, 0] * _inputWidth;
                            float y1 = boxesTensor[0, i, 1] * _inputHeight;
                            float x2 = boxesTensor[0, i, 2] * _inputWidth;
                            float y2 = boxesTensor[0, i, 3] * _inputHeight;

                            double w = Math.Max(0, x2 - x1);
                            double h = Math.Max(0, y2 - y1);

                            candidateRects.Add(new Rect2d(x1, y1, w, h));
                            candidateScores.Add(faceScore);
                        }
                    }
                }

                if (candidateRects.Count == 0)
                {
                    return 0;
                }

                // Filter overlapping anchor predictions using custom Non-Maximum Suppression (NMS)
                return CountFacesWithNMS(candidateRects, candidateScores, 0.4f);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error during GPU ONNX inference: {ex.Message}");
                return 0;
            }
        }

        private static int CountFacesWithNMS(List<Rect2d> boxes, List<float> scores, float iouThreshold)
        {
            var indices = Enumerable.Range(0, boxes.Count)
                .OrderByDescending(i => scores[i])
                .ToList();

            int faceCount = 0;
            while (indices.Count > 0)
            {
                int current = indices[0];
                faceCount++;
                indices.RemoveAt(0);

                for (int i = indices.Count - 1; i >= 0; i--)
                {
                    int other = indices[i];
                    var intersect = boxes[current].Intersect(boxes[other]);
                    double iou = 0.0;
                    if (intersect.Width > 0 && intersect.Height > 0)
                    {
                        double intersectArea = intersect.Width * intersect.Height;
                        double unionArea = (boxes[current].Width * boxes[current].Height) + 
                                           (boxes[other].Width * boxes[other].Height) - intersectArea;
                        if (unionArea > 0)
                        {
                            iou = intersectArea / unionArea;
                        }
                    }
                    if (iou > iouThreshold)
                    {
                        indices.RemoveAt(i);
                    }
                }
            }
            return faceCount;
        }

        public void Dispose()
        {
            _session?.Dispose();
            _session = null;
            GC.SuppressFinalize(this);
        }
    }
}
