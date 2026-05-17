import { Component, OnInit, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AssetService } from '../../core/services/asset';

@Component({
  selector: 'app-watchlist',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './watchlist.html',
  styleUrl: './watchlist.scss',
})
export class Watchlist implements OnInit {
  assets = signal<any[]>([]);
  watchlist: any[] = [];
  loading = true;
  actionLoading: Record<string, boolean> = {};
  searchQuery = signal<string>('');

  filteredAssets = computed(() => {
    const q = this.searchQuery().trim().toLowerCase();
    const all = this.assets();
    if (!q) return all;
    return all.filter(
      (a) =>
        (a.symbol ?? '').toLowerCase().includes(q) ||
        (a.name ?? '').toLowerCase().includes(q),
    );
  });

  constructor(private assetService: AssetService) {}

  ngOnInit(): void {
    this.assetService.getAssets().subscribe({
      next: (a) => this.assets.set(a),
    });
    this.assetService.getWatchlist().subscribe({
      next: (w) => {
        this.watchlist = w;
        this.loading = false;
      },
      error: () => (this.loading = false),
    });
  }

  onSearchInput(event: Event): void {
    this.searchQuery.set((event.target as HTMLInputElement).value);
  }

  isWatched(assetId: string): boolean {
    return this.watchlist.some((w) => w.asset.id === assetId);
  }

  watchlistEntryId(assetId: string): string | undefined {
    return this.watchlist.find((w) => w.asset.id === assetId)?.id;
  }

  toggle(asset: any): void {
    if (this.actionLoading[asset.id]) return;
    this.actionLoading[asset.id] = true;

    if (this.isWatched(asset.id)) {
      const entryId = this.watchlistEntryId(asset.id)!;
      this.assetService.removeFromWatchlist(entryId).subscribe({
        next: () => {
          this.watchlist = this.watchlist.filter((w) => w.id !== entryId);
          this.actionLoading[asset.id] = false;
        },
        error: () => (this.actionLoading[asset.id] = false),
      });
    } else {
      this.assetService.addToWatchlist(asset.id).subscribe({
        next: (entry) => {
          this.watchlist = [...this.watchlist, entry];
          this.actionLoading[asset.id] = false;
        },
        error: () => (this.actionLoading[asset.id] = false),
      });
    }
  }
}
