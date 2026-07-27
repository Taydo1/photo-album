using PhotoApp2.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PhotoApp2.Services
{
    public class AlbumPage
    {
        public int PageNumber { get; set; }
        public string Theme { get; set; } = string.Empty;
        public List<PhotoItem> Photos { get; set; } = new();
        public DateTime StartDate => Photos.Any() ? Photos.Min(p => p.DateTaken) : DateTime.MinValue;
        public DateTime EndDate => Photos.Any() ? Photos.Max(p => p.DateTaken) : DateTime.MinValue;
    }

    public class AlbumGeneratorService
    {
        private readonly Random _rand = new();

        public Task<List<AlbumPage>> GenerateAlbumAsync(IEnumerable<PhotoItem> allPhotos, string destFolderPath, IProgress<string>? progress = null)
        {
            return Task.Run(() =>
            {
                progress?.Report("Filtering and clustering photos by holiday trips and events...");

                // 1. Filter out unanalyzed or blurry photos (unless marked as Favorite)
                var candidates = allPhotos.Where(p => (p.IsAnalyzed && p.SharpnessScore > 50) || p.IsFavorite)
                                          .OrderBy(p => p.DateTaken)
                                          .ToList();

                if (!candidates.Any())
                    return new List<AlbumPage>();

                // 2. Chronological Episode Clustering (Same day / week or holidays)
                // A gap of > 48 hours indicates a distinct event or trip episode
                var episodes = new List<List<PhotoItem>>();
                var currentEpisode = new List<PhotoItem>();
                DateTime lastDate = DateTime.MinValue;

                foreach (var photo in candidates)
                {
                    if (lastDate != DateTime.MinValue && (photo.DateTaken - lastDate).TotalHours > 48)
                    {
                        if (currentEpisode.Any()) episodes.Add(currentEpisode);
                        currentEpisode = new List<PhotoItem>();
                    }
                    currentEpisode.Add(photo);
                    lastDate = photo.DateTaken;
                }
                if (currentEpisode.Any()) episodes.Add(currentEpisode);

                progress?.Report($"Identified {episodes.Count} chronological episodes. Selecting best photos and removing rapid bursts...");

                // 3. Quality scoring & rapid burst deduplication (within 15-second rolling window)
                var curatedPhotosByEpisode = new List<List<PhotoItem>>();
                foreach (var episode in episodes)
                {
                    var deduplicated = DeduplicateBursts(episode, 15.0);

                    // Sort by composite score to select top quality shots per episode
                    var scored = deduplicated.OrderByDescending(CalculatePhotoScore).ToList();

                    // If episode has many photos, take top shots (max ~100 per episode) while keeping diversity
                    int quota = Math.Min(scored.Count, Math.Max(10, scored.Count * 8 / 10));
                    var selected = scored.Take(quota).OrderBy(p => p.DateTaken).ToList();

                    if (selected.Any()) curatedPhotosByEpisode.Add(selected);
                }

                progress?.Report("Creating thematic pages (between 2 and 9 photos per page, max 1-2 photo kinds)...");

                // 4. Page pagination enforcing between 2-9 photos (weighted to 3-6) and max 1 or 2 kinds per page
                var pages = new List<AlbumPage>();
                int pageCounter = 1;

                foreach (var episodePhotos in curatedPhotosByEpisode)
                {
                    var episodePages = PaginateEpisode(episodePhotos, ref pageCounter);
                    pages.AddRange(episodePages);
                }

                progress?.Report($"Exporting {pages.Count} album pages to {destFolderPath}...");

                // 5. Folder export with names "Page_001_<theme>" and visual HTML preview catalog
                Directory.CreateDirectory(destFolderPath);
                foreach (var page in pages)
                {
                    string safeTheme = SanitizeFileName(page.Theme);
                    string pageFolderName = $"Page_{page.PageNumber:D3}_{safeTheme}";
                    string pageFolderPath = Path.Combine(destFolderPath, pageFolderName);
                    Directory.CreateDirectory(pageFolderPath);

                    int photoIndex = 1;
                    foreach (var photo in page.Photos)
                    {
                        string destPath = Path.Combine(pageFolderPath, $"{page.PageNumber:D3}_{photoIndex:D2}_{photo.FileName}");
                        try
                        {
                            if (!File.Exists(destPath) && File.Exists(photo.FilePath))
                            {
                                File.Copy(photo.FilePath, destPath);
                            }
                        }
                        catch (Exception ex)
                        {
                            Debug.WriteLine($"Failed copying file {photo.FilePath} to {destPath}: {ex.Message}");
                        }
                        photoIndex++;
                    }
                }

                GenerateHtmlPreview(pages, destFolderPath);

                progress?.Report($"Album generation complete! Created {pages.Count} pages in {destFolderPath}.");
                return pages;
            });
        }

        private List<PhotoItem> DeduplicateBursts(List<PhotoItem> sortedPhotos, double windowSeconds)
        {
            if (sortedPhotos.Count <= 1) return sortedPhotos;

            var result = new List<PhotoItem>();
            var burstWindow = new List<PhotoItem>();
            DateTime lastTimestamp = sortedPhotos.First().DateTaken;

            foreach (var photo in sortedPhotos)
            {
                if ((photo.DateTaken - lastTimestamp).TotalSeconds <= windowSeconds)
                {
                    burstWindow.Add(photo);
                }
                else
                {
                    // Pick the highest scoring shot from the closed burst window
                    result.Add(burstWindow.OrderByDescending(CalculatePhotoScore).First());
                    burstWindow.Clear();
                    burstWindow.Add(photo);
                }
                lastTimestamp = photo.DateTaken;
            }

            if (burstWindow.Any())
            {
                result.Add(burstWindow.OrderByDescending(CalculatePhotoScore).First());
            }

            return result;
        }

        private static double CalculatePhotoScore(PhotoItem p)
        {
            double score = p.SharpnessScore + (p.FaceCount * 200);
            if (p.IsFavorite) score += 2000;

            if (!string.IsNullOrEmpty(p.Keywords))
            {
                if (p.Keywords.Contains("beautiful landscape", StringComparison.OrdinalIgnoreCase)) score += 300;
                if (p.Keywords.Contains("architecture", StringComparison.OrdinalIgnoreCase) || 
                    p.Keywords.Contains("landmark", StringComparison.OrdinalIgnoreCase)) score += 200;
                if (p.Keywords.Contains("sunset", StringComparison.OrdinalIgnoreCase) || 
                    p.Keywords.Contains("waterfront", StringComparison.OrdinalIgnoreCase)) score += 200;
            }

            return score;
        }

        private List<AlbumPage> PaginateEpisode(List<PhotoItem> episodePhotos, ref int pageCounter)
        {
            var pages = new List<AlbumPage>();
            var currentPagePhotos = new List<PhotoItem>();
            int targetCapacity = PickNextPageCapacity();

            foreach (var photo in episodePhotos)
            {
                // Enforce max 1 or 2 kinds per page
                var tentativeKinds = currentPagePhotos.Select(p => p.PrimaryKind)
                                                      .Concat(new[] { photo.PrimaryKind })
                                                      .Where(k => !string.IsNullOrEmpty(k))
                                                      .Distinct()
                                                      .ToList();

                bool exceedsKinds = tentativeKinds.Count > 2;
                bool reachesCap = currentPagePhotos.Count >= targetCapacity;

                if (currentPagePhotos.Any() && (reachesCap || exceedsKinds))
                {
                    pages.Add(CreateAlbumPage(currentPagePhotos, pageCounter++));
                    currentPagePhotos = new List<PhotoItem>();
                    targetCapacity = PickNextPageCapacity();
                }

                currentPagePhotos.Add(photo);
            }

            if (currentPagePhotos.Any())
            {
                // If the last trailing page only has 1 photo, try merging it with the previous page if total <= 9 and <= 2 kinds
                if (currentPagePhotos.Count == 1 && pages.Any())
                {
                    var prevPage = pages.Last();
                    var combinedKinds = prevPage.Photos.Select(p => p.PrimaryKind)
                                                 .Concat(currentPagePhotos.Select(p => p.PrimaryKind))
                                                 .Where(k => !string.IsNullOrEmpty(k))
                                                 .Distinct()
                                                 .Count();

                    if (prevPage.Photos.Count < 9 && combinedKinds <= 2)
                    {
                        prevPage.Photos.AddRange(currentPagePhotos);
                        prevPage.Theme = DeterminePageTheme(prevPage.Photos);
                        currentPagePhotos.Clear();
                    }
                }

                if (currentPagePhotos.Any())
                {
                    pages.Add(CreateAlbumPage(currentPagePhotos, pageCounter++));
                }
            }

            return pages;
        }

        private int PickNextPageCapacity()
        {
            // Pick between 2 and 9, with much higher probability to 3-6
            // Distribution: 2 (10%), 3 (20%), 4 (25%), 5 (20%), 6 (15%), 7 (4%), 8 (3%), 9 (3%) -> 3-6 is 80%
            int roll = _rand.Next(100);
            if (roll < 10) return 2;
            if (roll < 30) return 3;
            if (roll < 55) return 4;
            if (roll < 75) return 5;
            if (roll < 90) return 6;
            if (roll < 94) return 7;
            if (roll < 97) return 8;
            return 9;
        }

        private static AlbumPage CreateAlbumPage(List<PhotoItem> photos, int pageNum)
        {
            var page = new AlbumPage
            {
                PageNumber = pageNum,
                Photos = new List<PhotoItem>(photos),
                Theme = DeterminePageTheme(photos)
            };
            return page;
        }

        private static string DeterminePageTheme(List<PhotoItem> photos)
        {
            var kinds = photos.Select(p => p.PrimaryKind)
                              .Where(k => !string.IsNullOrEmpty(k))
                              .Distinct()
                              .OrderBy(k => k)
                              .ToList();

            if (!kinds.Any())
                return "Collection";

            return string.Join("_and_", kinds.Select(k => k.Replace(" ", "_")));
        }

        private static string SanitizeFileName(string name)
        {
            var invalid = Path.GetInvalidFileNameChars();
            var sb = new StringBuilder();
            foreach (var ch in name)
            {
                if (!invalid.Contains(ch) && ch != '/' && ch != '\\')
                    sb.Append(ch);
                else
                    sb.Append('_');
            }
            return sb.ToString().Trim('_');
        }

        private static void GenerateHtmlPreview(List<AlbumPage> pages, string destFolderPath)
        {
            var html = new StringBuilder();
            html.AppendLine("<!DOCTYPE html>");
            html.AppendLine("<html lang='en'><head><meta charset='utf-8'><title>Intelligent Photo Album Preview</title>");
            html.AppendLine("<style>");
            html.AppendLine(":root { --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #38bdf8; --border: #334155; }");
            html.AppendLine("body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 40px 20px; }");
            html.AppendLine(".container { max-width: 1200px; margin: 0 auto; }");
            html.AppendLine("h1 { text-align: center; font-weight: 700; color: var(--accent); margin-bottom: 10px; font-size: 2.5em; letter-spacing: -0.5px; }");
            html.AppendLine(".summary { text-align: center; color: #94a3b8; margin-bottom: 40px; font-size: 1.1em; }");
            html.AppendLine(".page-card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 25px; margin-bottom: 35px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); transition: transform 0.2s; }");
            html.AppendLine(".page-card:hover { transform: translateY(-3px); }");
            html.AppendLine(".page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 15px; }");
            html.AppendLine(".page-title { font-size: 1.5em; font-weight: 600; color: var(--text); display: flex; align-items: center; gap: 12px; }");
            html.AppendLine(".badge { background: var(--accent); color: #000; padding: 4px 12px; border-radius: 20px; font-size: 0.75em; font-weight: 700; text-transform: uppercase; }");
            html.AppendLine(".page-meta { color: #94a3b8; font-size: 0.9em; }");
            html.AppendLine(".photo-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 15px; }");
            html.AppendLine(".photo-item { background: #000; border-radius: 10px; overflow: hidden; position: relative; aspect-ratio: 4/3; }");
            html.AppendLine(".photo-item img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s ease; }");
            html.AppendLine(".photo-item:hover img { transform: scale(1.05); }");
            html.AppendLine(".photo-tag { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0, 0, 0, 0.75); backdrop-filter: blur(4px); color: #e2e8f0; font-size: 0.8em; padding: 8px 12px; box-sizing: border-box; }");
            html.AppendLine("</style></head><body>");
            html.AppendLine("<div class='container'>");
            html.AppendLine("<h1>✨ AI Curated Photo Album</h1>");
            html.AppendLine($"<div class='summary'>Generated on {DateTime.Now:MMMM dd, yyyy} &bull; {pages.Count} Chronological & Thematic Pages &bull; {pages.Sum(p => p.Photos.Count)} Total Photos</div>");

            foreach (var page in pages)
            {
                string safeTheme = SanitizeFileName(page.Theme);
                string pageFolderName = $"Page_{page.PageNumber:D3}_{safeTheme}";
                string readableTheme = page.Theme.Replace("_and_", " & ").Replace("_", " ");

                html.AppendLine("<div class='page-card'>");
                html.AppendLine("<div class='page-header'>");
                html.AppendLine($"<div class='page-title'>Page {page.PageNumber} <span class='badge'>{readableTheme}</span></div>");
                string dateStr = page.StartDate.Date == page.EndDate.Date 
                    ? page.StartDate.ToString("MMM dd, yyyy") 
                    : $"{page.StartDate:MMM dd} &ndash; {page.EndDate:MMM dd, yyyy}";
                html.AppendLine($"<div class='page-meta'>{dateStr} &bull; {page.Photos.Count} Photos</div>");
                html.AppendLine("</div>");
                html.AppendLine("<div class='photo-grid'>");

                int pIdx = 1;
                foreach (var photo in page.Photos)
                {
                    string relPath = $"{pageFolderName}/{page.PageNumber:D3}_{pIdx:D2}_{photo.FileName}";
                    string caption = string.IsNullOrEmpty(photo.Keywords) ? photo.FileName : photo.Keywords;
                    html.AppendLine($"<div class='photo-item' title='{photo.FileName} &bull; {caption}'>");
                    html.AppendLine($"<img src='{relPath}' alt='{photo.FileName}' loading='lazy'/>");
                    html.AppendLine($"<div class='photo-tag'>{caption}</div>");
                    html.AppendLine("</div>");
                    pIdx++;
                }

                html.AppendLine("</div></div>");
            }

            html.AppendLine("</div></body></html>");
            try
            {
                File.WriteAllText(Path.Combine(destFolderPath, "Album_Preview.html"), html.ToString(), Encoding.UTF8);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Failed to write HTML preview: {ex.Message}");
            }
        }
    }
}
