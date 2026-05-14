from rest_framework import serializers
from .models import Asset, Watchlist


class AssetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Asset
        fields = ('id', 'symbol', 'name', 'market_cap', 'category', 'is_active')


class WatchlistSerializer(serializers.ModelSerializer):
    asset = AssetSerializer(read_only=True)
    asset_id = serializers.UUIDField(write_only=True)

    class Meta:
        model = Watchlist
        fields = ('id', 'asset', 'asset_id', 'added_at')

    def validate_asset_id(self, value):
        if not Asset.objects.filter(id=value, is_active=True).exists():
            raise serializers.ValidationError('Geçersiz varlık.')
        return value

    def create(self, validated_data):
        asset = Asset.objects.get(id=validated_data['asset_id'])
        user = self.context['request'].user
        if Watchlist.objects.filter(user=user, asset=asset).exists():
            raise serializers.ValidationError('Bu varlık zaten watchlist\'te.')
        return Watchlist.objects.create(user=user, asset=asset)
