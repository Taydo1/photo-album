using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Input;
using PhotoApp2.ViewModels;
using System;
using WinRT.Interop;
using System.ComponentModel;
using PhotoApp2.Models;
using Windows.Storage.Pickers;
using Windows.System;

namespace PhotoApp2
{
    public sealed partial class MainWindow : Window, INotifyPropertyChanged
    {
        public MainViewModel ViewModel { get; }

        public event PropertyChangedEventHandler? PropertyChanged;

        public MainWindow()
        {
            InitializeComponent();
            ViewModel = new MainViewModel(App.Database, App.Analyzer);
            
            ViewModel.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ViewModel.SelectedPhoto))
                {
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(ViewModel.SelectedPhoto)));
                }
            };
        }

        private async void OpenFolder_Click(object sender, RoutedEventArgs e)
        {
            var hwnd = WindowNative.GetWindowHandle(this);
            var folderPicker = new FolderPicker();
            InitializeWithWindow.Initialize(folderPicker, hwnd);
            folderPicker.FileTypeFilter.Add("*");

            var folder = await folderPicker.PickSingleFolderAsync();
            if (folder != null)
            {
                await ViewModel.OpenFolderAsync(folder.Path);
            }
        }

        private void AnalyzePhotos_Click(object sender, RoutedEventArgs e)
        {
            ViewModel.AnalyzePhotosCommand.Execute(null);
        }

        private async void AutoGenerateAlbum_Click(object sender, RoutedEventArgs e)
        {
            var hwnd = WindowNative.GetWindowHandle(this);
            var folderPicker = new FolderPicker();
            InitializeWithWindow.Initialize(folderPicker, hwnd);
            folderPicker.FileTypeFilter.Add("*");

            var destFolder = await folderPicker.PickSingleFolderAsync();
            if (destFolder != null)
            {
                await ViewModel.ExecuteAutoGenerateAlbumAsync(destFolder.Path);
            }
        }

        private async void FavoriteToggle_Click(object sender, RoutedEventArgs e)
        {
            if (ViewModel.SelectedPhoto != null)
            {
                await App.Database.UpdateFavoriteStatusAsync(ViewModel.SelectedPhoto.FilePath, ViewModel.SelectedPhoto.IsFavorite);
            }
        }

        private void ExportFavorites_Click(object sender, RoutedEventArgs e)
        {
            var hwnd = WindowNative.GetWindowHandle(this);
            ViewModel.ExportFavoritesCommand.Execute(hwnd);
        }

        private async void Grid_KeyDown(object sender, KeyRoutedEventArgs e)
        {
            if (e.Key == VirtualKey.Space && ViewModel.SelectedPhoto != null)
            {
                ViewModel.SelectedPhoto.IsFavorite = !ViewModel.SelectedPhoto.IsFavorite;
                await App.Database.UpdateFavoriteStatusAsync(ViewModel.SelectedPhoto.FilePath, ViewModel.SelectedPhoto.IsFavorite);
                e.Handled = true;
            }
        }
    }
}
