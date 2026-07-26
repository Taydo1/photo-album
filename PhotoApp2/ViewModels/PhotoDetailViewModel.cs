using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PhotoApp2.Models;
using PhotoApp2.Services;

namespace PhotoApp2.ViewModels
{
    public partial class PhotoDetailViewModel : ObservableObject
    {
        private readonly DatabaseService _dbService;

        [ObservableProperty]
        private PhotoItem? _photo;

        public PhotoDetailViewModel(DatabaseService dbService)
        {
            _dbService = dbService;
        }

        [RelayCommand]
        private async Task ToggleFavoriteAsync()
        {
            if (Photo != null)
            {
                Photo.IsFavorite = !Photo.IsFavorite;
                await _dbService.UpdateFavoriteStatusAsync(Photo.FilePath, Photo.IsFavorite);
            }
        }
    }
}
