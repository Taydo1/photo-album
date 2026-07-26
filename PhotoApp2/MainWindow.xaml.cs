using Microsoft.UI.Xaml;
using PhotoApp2.ViewModels;
using System;
using WinRT.Interop;
using System.ComponentModel;
using PhotoApp2.Models;

namespace PhotoApp2
{
    public sealed partial class MainWindow : Window, INotifyPropertyChanged
    {
        public MainViewModel ViewModel { get; }

        public event PropertyChangedEventHandler? PropertyChanged;

        public bool IsPhotoSelected => ViewModel.SelectedPhoto != null;

        public MainWindow()
        {
            InitializeComponent();
            ViewModel = new MainViewModel(App.Database, App.Analyzer);
            
            ViewModel.PropertyChanged += (s, e) =>
            {
                if (e.PropertyName == nameof(ViewModel.SelectedPhoto))
                {
                    PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IsPhotoSelected)));
                }
            };

            // Load photos initially
            _ = ViewModel.LoadPhotosAsync();
        }

        private void ImportFolder_Click(object sender, RoutedEventArgs e)
        {
            var hwnd = WindowNative.GetWindowHandle(this);
            ViewModel.ImportFolderCommand.Execute(hwnd);
        }

        private void ExportFavorites_Click(object sender, RoutedEventArgs e)
        {
            var hwnd = WindowNative.GetWindowHandle(this);
            ViewModel.ExportFavoritesCommand.Execute(hwnd);
        }

        private void AutoGenerateAlbum_Click(object sender, RoutedEventArgs e)
        {
            var hwnd = WindowNative.GetWindowHandle(this);
            ViewModel.AutoGenerateAlbumCommand.Execute(hwnd);
        }

        private async void FavoriteToggle_Click(object sender, RoutedEventArgs e)
        {
            if (ViewModel.SelectedPhoto != null)
            {
                // Because SelectedPhoto.IsFavorite is updated via TwoWay binding, 
                // we just need to persist it.
                await App.Database.UpdateFavoriteStatusAsync(ViewModel.SelectedPhoto.FilePath, ViewModel.SelectedPhoto.IsFavorite);
                
                // Force UI refresh if needed, but ObservableObject should handle it
            }
        }
    }
}
