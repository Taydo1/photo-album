using System;
using System.Collections.Generic;
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

        [Fact]
        public async Task AnalyzePhotoAsync_ExtractsKeywordsAndPrimaryKind_Successfully()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            var sampleFile = Directory.GetFiles(imagesDir, "*.jpg").FirstOrDefault();
            Assert.NotNull(sampleFile);

            var result = await service.AnalyzePhotoAsync(sampleFile);
            _output.WriteLine($"File: {result.FileName} | PrimaryKind: {result.PrimaryKind} | Keywords: {result.Keywords}");

            Assert.True(result.IsAnalyzed);
            Assert.False(string.IsNullOrEmpty(result.PrimaryKind), "PrimaryKind should not be empty");
        }

        [Fact]
        public async Task AlbumGeneratorService_EnforcesOneOrTwoKindsPerPage_AndFormatsFolderNames()
        {
            var generator = new AlbumGeneratorService();
            var tempDir = Path.Combine(Path.GetTempPath(), "PhotoApp2_TestAlbum_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempDir);

            try
            {
                // Create synthetic analyzed photos across distinct themes in the same holiday week
                var simPhotos = new List<Models.PhotoItem>
                {
                    new Models.PhotoItem { Id = 1, FilePath = "fake1.jpg", FileName = "fake1.jpg", IsAnalyzed = true, SharpnessScore = 100, DateTaken = DateTime.Today.AddHours(1), PrimaryKind = "Landscape", Keywords = "beautiful landscape" },
                    new Models.PhotoItem { Id = 2, FilePath = "fake2.jpg", FileName = "fake2.jpg", IsAnalyzed = true, SharpnessScore = 120, DateTaken = DateTime.Today.AddHours(2), PrimaryKind = "Person", Keywords = "person" },
                    new Models.PhotoItem { Id = 3, FilePath = "fake3.jpg", FileName = "fake3.jpg", IsAnalyzed = true, SharpnessScore = 110, DateTaken = DateTime.Today.AddHours(3), PrimaryKind = "Architecture", Keywords = "architecture" },
                    new Models.PhotoItem { Id = 4, FilePath = "fake4.jpg", FileName = "fake4.jpg", IsAnalyzed = true, SharpnessScore = 105, DateTaken = DateTime.Today.AddHours(4), PrimaryKind = "Animal", Keywords = "animal" }
                };

                var pages = await generator.GenerateAlbumAsync(simPhotos, tempDir);

                _output.WriteLine($"Generated {pages.Count} pages.");
                foreach (var page in pages)
                {
                    _output.WriteLine($"Page {page.PageNumber} Theme: {page.Theme} | Photos: {page.Photos.Count} | Distinct Kinds: {page.Photos.Select(p => p.PrimaryKind).Distinct().Count()}");
                    
                    // Verify max 1 or 2 kinds rule
                    var kindsOnPage = page.Photos.Select(p => p.PrimaryKind).Distinct().ToList();
                    Assert.True(kindsOnPage.Count <= 2, $"Page {page.PageNumber} exceeded 2 distinct kinds (found {kindsOnPage.Count})");

                    // Verify folder name formatting "Page_001_<theme>"
                    string expectedPrefix = $"Page_{page.PageNumber:D3}_";
                    var matchingFolders = Directory.GetDirectories(tempDir).Where(d => Path.GetFileName(d).StartsWith(expectedPrefix)).ToList();
                    Assert.NotEmpty(matchingFolders);
                }

                // Verify HTML preview creation
                var htmlPath = Path.Combine(tempDir, "Album_Preview.html");
                Assert.True(File.Exists(htmlPath), "Album_Preview.html should be created");
            }
            finally
            {
                if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
            }
        }
    }
}
