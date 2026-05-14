from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import Asset, Watchlist
from .serializers import AssetSerializer, WatchlistSerializer


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def asset_list(request):
    assets = Asset.objects.filter(is_active=True)
    serializer = AssetSerializer(assets, many=True)
    return Response(serializer.data)


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def watchlist(request):
    if request.method == 'GET':
        items = Watchlist.objects.filter(user=request.user).select_related('asset')
        serializer = WatchlistSerializer(items, many=True)
        return Response(serializer.data)

    serializer = WatchlistSerializer(data=request.data, context={'request': request})
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def watchlist_delete(request, pk):
    try:
        item = Watchlist.objects.get(id=pk, user=request.user)
        item.delete()
        return Response({'message': 'Watchlist\'ten kaldırıldı.'})
    except Watchlist.DoesNotExist:
        return Response({'error': 'Bulunamadı.'}, status=status.HTTP_404_NOT_FOUND)
