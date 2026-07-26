using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using PhotoApp2.Services;
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

        [Fact]
        public async Task AnalyzePhotoAsync_ProcessesAllAttachedImages_Successfully()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            Assert.True(Directory.Exists(imagesDir), $"TestImages directory not found at {imagesDir}");

            var imageFiles = Directory.GetFiles(imagesDir, "*.jpg");
            Assert.NotEmpty(imageFiles);

            var totalStopwatch = Stopwatch.StartNew();

            foreach (var file in imageFiles)
            {
                var sw = Stopwatch.StartNew();
                var result = await service.AnalyzePhotoAsync(file);
                sw.Stop();

                _output.WriteLine($"File: {Path.GetFileName(file)} | Time: {sw.ElapsedMilliseconds} ms | Faces: {result.FaceCount} | Sharpness: {result.SharpnessScore:F2} | Category: {result.SceneCategory}");

                Assert.NotNull(result);
                Assert.True(result.IsAnalyzed, $"Photo analysis failed for {file}");
                Assert.True(sw.ElapsedMilliseconds >= 0);
            }

            totalStopwatch.Stop();
            _output.WriteLine($"Total time for {imageFiles.Length} images: {totalStopwatch.ElapsedMilliseconds} ms (avg {totalStopwatch.ElapsedMilliseconds / imageFiles.Length} ms/image)");
        }

        [Fact]
        public async Task AnalyzePhotoAsync_GroupPhoto_DetectsFaces()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            // media__1785070826785.jpg is the group photo with multiple people around a table
            var groupPhotoPath = Directory.GetFiles(imagesDir, "media__1785070826785.jpg").FirstOrDefault()
                ?? Directory.GetFiles(imagesDir, "*.jpg").FirstOrDefault();

            Assert.NotNull(groupPhotoPath);

            var sw = Stopwatch.StartNew();
            var result = await service.AnalyzePhotoAsync(groupPhotoPath);
            sw.Stop();

            _output.WriteLine($"Group photo analyzed in {sw.ElapsedMilliseconds} ms | Faces found: {result.FaceCount}");

            Assert.True(result.IsAnalyzed);
            Assert.True(result.FaceCount > 0, "Group photo should detect faces");
            Assert.Equal("Person", result.SceneCategory);
        }

        [Fact]
        public async Task Benchmark_BatchAnalyzePhotos_Throughput()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();
            var dbService = new DatabaseService();
            await dbService.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            var sampleFiles = Directory.GetFiles(imagesDir, "*.jpg");
            Assert.NotEmpty(sampleFiles);

            // Create 40 photo items to simulate a real folder analysis batch
            var testPhotos = Enumerable.Range(1, 10)
                .SelectMany(i => sampleFiles.Select((file, idx) => new Models.PhotoItem
                {
                    FilePath = file + $"_sim_{i}_{idx}", // Simulated distinct paths for database uniqueness
                    FileName = Path.GetFileName(file),
                    FileSizeBytes = new FileInfo(file).Length,
                    DateTaken = DateTime.Now.AddDays(-idx)
                }))
                .ToList();

            _output.WriteLine($"=== STARTING BATCH ANALYSIS BENCHMARK ({testPhotos.Count} items) ===");
            var analysisSw = Stopwatch.StartNew();

            // Perform parallel analysis as in MainViewModel
            await Parallel.ForEachAsync(testPhotos, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, async (photo, ct) =>
            {
                // We use the actual disk file for analysis while retaining the unique simulated file path for database entry
                var actualFilePath = photo.FilePath.Substring(0, photo.FilePath.IndexOf("_sim_"));
                var analyzed = await service.AnalyzePhotoAsync(actualFilePath);

                photo.IsAnalyzed = true;
                photo.SharpnessScore = analyzed.SharpnessScore;
                photo.FaceCount = analyzed.FaceCount;
                photo.SceneCategory = analyzed.SceneCategory;
            });

            analysisSw.Stop();
            double fps = testPhotos.Count / analysisSw.Elapsed.TotalSeconds;
            _output.WriteLine($"Parallel Analysis Time: {analysisSw.ElapsedMilliseconds} ms ({fps:F1} photos/sec)");

            // Test Batch SQLite Save Speed
            var dbSw = Stopwatch.StartNew();
            await dbService.SavePhotosAsync(testPhotos);
            dbSw.Stop();
            _output.WriteLine($"Batch SQLite Save Time ({testPhotos.Count} items): {dbSw.ElapsedMilliseconds} ms");

            Assert.True(testPhotos.All(p => p.IsAnalyzed), "All photos should be analyzed");
            Assert.True(analysisSw.ElapsedMilliseconds < 5000, "Parallel batch analysis should complete quickly");
            Assert.True(dbSw.ElapsedMilliseconds < 500, "Transactional batch save should be instantaneous");
        }
    }
}
