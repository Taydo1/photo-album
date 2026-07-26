using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PhotoApp2.Models;
using PhotoApp2.Services;
using Windows.Storage.Pickers;
using WinRT.Interop;
using Windows.Storage;

namespace PhotoApp2.ViewModels
{
    public partial class MainViewModel : ObservableObject
    {
        private readonly DatabaseService _dbService;
        private readonly PhotoAnalyzerService _analyzerService;

        [ObservableProperty]
        private ObservableCollection<PhotoItem> _photos = new();

        [ObservableProperty]
        private bool _isAnalyzing;

        [ObservableProperty]
        private string _statusMessage = "Ready";

        [ObservableProperty]
        private int _totalPhotos;

        [ObservableProperty]
        private int _analyzedPhotos;
        
        [ObservableProperty]
        private PhotoItem? _selectedPhoto;

        // Filtering
        [ObservableProperty]
        private bool _showFavoritesOnly;

        [ObservableProperty]
        private bool _showPeopleOnly;
        
        private List<PhotoItem> _allPhotosCache = new();

        public MainViewModel(DatabaseService dbService, PhotoAnalyzerService analyzerService)
        {
            _dbService = dbService;
            _analyzerService = analyzerService;
        }

        public async Task LoadPhotosAsync()
        {
            StatusMessage = "Loading photos from database...";
            _allPhotosCache = await _dbService.GetAllPhotosAsync();
            ApplyFilters();
            StatusMessage = $"Loaded {_allPhotosCache.Count} photos.";
        }

        [RelayCommand]
        private async Task ImportFolderAsync(IntPtr hwnd)
        {
            var folderPicker = new FolderPicker();
            InitializeWithWindow.Initialize(folderPicker, hwnd);
            folderPicker.FileTypeFilter.Add("*");

            var folder = await folderPicker.PickSingleFolderAsync();
            if (folder != null)
            {
                await AnalyzeFolderAsync(folder.Path);
            }
        }

        private async Task AnalyzeFolderAsync(string folderPath)
        {
            IsAnalyzing = true;
            StatusMessage = $"Scanning {folderPath}...";

            var files = Directory.GetFiles(folderPath, "*.*", SearchOption.TopDirectoryOnly)
                .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".heic", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".cr2", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".nef", StringComparison.OrdinalIgnoreCase))
                .ToList();

            TotalPhotos = files.Count;
            AnalyzedPhotos = 0;
            StatusMessage = $"Found {TotalPhotos} photos. Analyzing...";

            await _analyzerService.InitializeAsync();

            var newPhotos = new List<PhotoItem>();

            foreach (var file in files)
            {
                // Check if already in DB
                var existing = _allPhotosCache.FirstOrDefault(p => p.FilePath.Equals(file, StringComparison.OrdinalIgnoreCase));
                if (existing != null && existing.IsAnalyzed)
                {
                    AnalyzedPhotos++;
                    continue;
                }

                var photo = await _analyzerService.AnalyzePhotoAsync(file);
                await _dbService.SavePhotoAsync(photo);
                
                // Update cache
                if (existing != null)
                {
                    _allPhotosCache.Remove(existing);
                }
                _allPhotosCache.Add(photo);

                AnalyzedPhotos++;
                StatusMessage = $"Analyzing... {AnalyzedPhotos}/{TotalPhotos}";
            }

            IsAnalyzing = false;
            await LoadPhotosAsync();
        }

        partial void OnShowFavoritesOnlyChanged(bool value) => ApplyFilters();
        partial void OnShowPeopleOnlyChanged(bool value) => ApplyFilters();

        private void ApplyFilters()
        {
            var filtered = _allPhotosCache.AsEnumerable();
            if (ShowFavoritesOnly)
            {
                filtered = filtered.Where(p => p.IsFavorite);
            }
            if (ShowPeopleOnly)
            {
                filtered = filtered.Where(p => p.FaceCount > 0);
            }
            
            Photos = new ObservableCollection<PhotoItem>(filtered.OrderBy(p => p.DateTaken));
        }

        [RelayCommand]
        private async Task ExportFavoritesAsync(IntPtr hwnd)
        {
            var favorites = _allPhotosCache.Where(p => p.IsFavorite).ToList();
            if (!favorites.Any()) return;

            var folderPicker = new FolderPicker();
            InitializeWithWindow.Initialize(folderPicker, hwnd);
            folderPicker.FileTypeFilter.Add("*");

            var destFolder = await folderPicker.PickSingleFolderAsync();
            if (destFolder == null) return;

            StatusMessage = "Exporting favorites...";
            int count = 0;
            foreach (var photo in favorites)
            {
                var destPath = Path.Combine(destFolder.Path, photo.FileName);
                if (!File.Exists(destPath))
                {
                    File.Copy(photo.FilePath, destPath);
                }
                count++;
            }
            StatusMessage = $"Exported {count} photos to {destFolder.Name}.";
        }

        [RelayCommand]
        private async Task AutoGenerateAlbumAsync(IntPtr hwnd)
        {
            if (!_allPhotosCache.Any()) return;

            var folderPicker = new FolderPicker();
            InitializeWithWindow.Initialize(folderPicker, hwnd);
            folderPicker.FileTypeFilter.Add("*");

            var destFolder = await folderPicker.PickSingleFolderAsync();
            if (destFolder == null) return;

            StatusMessage = "Generating album pages...";
            
            // 1. Filter out blurry or unanalyzed photos
            var candidates = _allPhotosCache.Where(p => p.IsAnalyzed && p.SharpnessScore > 100).ToList();

            // 2. Sort by date
            candidates = candidates.OrderBy(p => p.DateTaken).ToList();

            // 3. Simple clustering by time (e.g. gap > 4 hours means new event)
            var events = new List<List<PhotoItem>>();
            var currentEvent = new List<PhotoItem>();
            DateTime lastDate = DateTime.MinValue;

            foreach (var photo in candidates)
            {
                if (lastDate == DateTime.MinValue || (photo.DateTaken - lastDate).TotalHours > 4)
                {
                    currentEvent = new List<PhotoItem>();
                    events.Add(currentEvent);
                }
                currentEvent.Add(photo);
                lastDate = photo.DateTaken;
            }

            // 4. Select top photos per event (target ~400 total)
            int targetTotal = 400;
            int selectedTotal = 0;
            var selectedPhotos = new List<PhotoItem>();

            foreach (var ev in events)
            {
                // Assign a quota based on event size, but minimum 1
                int quota = Math.Max(1, (int)((double)ev.Count / candidates.Count * targetTotal));
                
                // Sort by a heuristic score: Sharpness + Faces (we weight faces heavily)
                var bestInEvent = ev.OrderByDescending(p => p.SharpnessScore + (p.FaceCount * 500))
                                    .Take(quota)
                                    .OrderBy(p => p.DateTaken) // Re-sort by time
                                    .ToList();
                
                selectedPhotos.AddRange(bestInEvent);
                selectedTotal += bestInEvent.Count;
            }

            // 5. Group into pages of 3 to 9 photos
            var pages = new List<List<PhotoItem>>();
            var currentPage = new List<PhotoItem>();
            
            foreach (var photo in selectedPhotos)
            {
                currentPage.Add(photo);
                // Simple heuristic: if we have 6 photos, maybe start a new page
                // Or if there's a large time gap
                if (currentPage.Count >= 6)
                {
                    pages.Add(currentPage);
                    currentPage = new List<PhotoItem>();
                }
            }
            if (currentPage.Any()) pages.Add(currentPage);

            // 6. Export the pages to folders
            int pageIndex = 1;
            foreach (var page in pages)
            {
                var pageFolder = await destFolder.CreateFolderAsync($"Page_{pageIndex:D3}", Windows.Storage.CreationCollisionOption.OpenIfExists);
                int photoIndex = 1;
                foreach (var photo in page)
                {
                    var destPath = Path.Combine(pageFolder.Path, $"{pageIndex:D3}_{photoIndex:D2}_{photo.FileName}");
                    if (!File.Exists(destPath))
                    {
                        File.Copy(photo.FilePath, destPath);
                    }
                    photoIndex++;
                }
                pageIndex++;
            }

            StatusMessage = $"Generated {pages.Count} pages with {selectedPhotos.Count} total photos.";
        }
    }
}
