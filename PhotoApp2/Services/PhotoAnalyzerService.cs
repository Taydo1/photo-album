using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using OpenCvSharp;
using Windows.Graphics.Imaging;
using Windows.Media.FaceAnalysis;
using Windows.Storage;
using PhotoApp2.Models;
using ImageMagick;

namespace PhotoApp2.Services
{
    public class PhotoAnalyzerService
    {
        private FaceDetector? _faceDetector;

        public async Task InitializeAsync()
        {
            if (_faceDetector == null)
            {
                _faceDetector = await FaceDetector.CreateAsync();
            }
        }

        public async Task<PhotoItem> AnalyzePhotoAsync(string filePath)
        {
            var fileInfo = new FileInfo(filePath);
            var photo = new PhotoItem
            {
                FilePath = filePath,
                FileName = fileInfo.Name,
                FileSizeBytes = fileInfo.Length,
                DateTaken = fileInfo.CreationTime, // Better to read EXIF, but this is a fallback
            };

            try
            {
                // Ensure ImageMagick can read it (handles JPG, RAW, HEIC)
                using var magickImage = new MagickImage(filePath);
                
                // Try to get actual DateTaken from EXIF
                var exifProfile = magickImage.GetExifProfile();
                if (exifProfile != null)
                {
                    var dateTakenTag = exifProfile.GetValue(ExifTag.DateTimeOriginal);
                    if (dateTakenTag != null && DateTime.TryParseExact(dateTakenTag.ToString(), "yyyy:MM:dd HH:mm:ss", null, System.Globalization.DateTimeStyles.None, out var dateTaken))
                    {
                        photo.DateTaken = dateTaken;
                    }
                }

                // 1. Convert to Bitmap for Windows FaceDetector and OpenCV
                // We convert it to a standard Jpeg format in memory to ensure compatibility
                using var memStream = new MemoryStream();
                magickImage.Format = MagickFormat.Jpeg;
                magickImage.Write(memStream);
                memStream.Position = 0;

                // 2. Sharpness Score (Variance of Laplacian) using OpenCV
                using (var mat = Cv2.ImDecode(memStream.ToArray(), ImreadModes.Grayscale))
                {
                    using var laplacian = new Mat();
                    Cv2.Laplacian(mat, laplacian, MatType.CV_64F);
                    Cv2.MeanStdDev(laplacian, out var mean, out var stddev);
                    var variance = stddev.Val0 * stddev.Val0;
                    photo.SharpnessScore = variance;
                }

                // 3. Face Detection
                memStream.Position = 0;
                var decoder = await BitmapDecoder.CreateAsync(memStream.AsRandomAccessStream());
                // Decode to Bgra8 which is universally supported and has no dimension restrictions
                using (var bgraBitmap = await decoder.GetSoftwareBitmapAsync(BitmapPixelFormat.Bgra8, BitmapAlphaMode.Ignore))
                {
                    // FaceDetector expects Gray8 or Nv12. Convert to Gray8 to avoid odd-dimension Nv12 errors.
                    using (var grayBitmap = SoftwareBitmap.Convert(bgraBitmap, BitmapPixelFormat.Gray8))
                    {
                        if (_faceDetector != null && grayBitmap != null)
                        {
                            var faces = await _faceDetector.DetectFacesAsync(grayBitmap);
                            photo.FaceCount = faces.Count;
                        }
                    }
                }

                // 4. Scene Classification (Heuristics / Placeholder for ONNX)
                // For a full implementation, we'd run an ONNX model here (e.g. ResNet Places365).
                // As a fallback/heuristic:
                if (photo.FaceCount > 0)
                {
                    photo.SceneCategory = "Person";
                }
                else
                {
                    // Fallback heuristics based on color/edges could go here, 
                    // but we will default to Landscape/Other for now until ONNX is integrated.
                    photo.SceneCategory = "Landscape / Other";
                }

                photo.IsAnalyzed = true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error analyzing photo {filePath}: {ex.Message}");
                photo.IsAnalyzed = false;
            }

            return photo;
        }
    }
}
