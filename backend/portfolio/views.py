from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import PortfolioJob
from .serializers import PortfolioJobSerializer, PortfolioJobDetailSerializer
from .tasks import run_portfolio_optimization


@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def job_list(request):
    if request.method == 'GET':
        jobs = PortfolioJob.objects.filter(user=request.user)
        serializer = PortfolioJobSerializer(jobs, many=True)
        return Response(serializer.data)

    params = request.data.get('params', {})
    if params.get('lookback_days', 365) < 250:
        return Response(
            {'error': 'lookback_days en az 250 olmalıdır.'},
            status=status.HTTP_400_BAD_REQUEST,
        )

    job = PortfolioJob.objects.create(
        user=request.user,
        params=params,
    )
    run_portfolio_optimization.delay(str(job.id), job.params)
    serializer = PortfolioJobSerializer(job)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def job_detail(request, pk):
    try:
        job = PortfolioJob.objects.get(id=pk, user=request.user)
        serializer = PortfolioJobDetailSerializer(job)
        return Response(serializer.data)
    except PortfolioJob.DoesNotExist:
        return Response({'error': 'Bulunamadı.'}, status=status.HTTP_404_NOT_FOUND)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def job_delete(request, pk):
    try:
        job = PortfolioJob.objects.get(id=pk, user=request.user)
        job.delete()
        return Response({'message': 'Job silindi.'})
    except PortfolioJob.DoesNotExist:
        return Response({'error': 'Bulunamadı.'}, status=status.HTTP_404_NOT_FOUND)
