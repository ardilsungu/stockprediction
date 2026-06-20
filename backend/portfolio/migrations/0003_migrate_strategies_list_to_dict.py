import re
from django.db import migrations


def _to_snake_case(value):
    value = re.sub(r'[^0-9a-zA-Z]+', '_', value).strip('_').lower()
    return value or 'strategy'


def list_to_dict(apps, schema_editor):
    PortfolioResult = apps.get_model('portfolio', 'PortfolioResult')
    for result in PortfolioResult.objects.all():
        strategies = result.strategies
        if not isinstance(strategies, list):
            continue

        converted = {}
        for idx, entry in enumerate(strategies):
            if not isinstance(entry, dict):
                continue
            key = entry.get('key')
            if not key:
                name = entry.get('name')
                key = _to_snake_case(name) if name else f'strategy_{idx}'
            payload = {k: v for k, v in entry.items() if k != 'key'}
            converted[key] = payload

        result.strategies = converted
        result.save(update_fields=['strategies'])


def dict_to_list(apps, schema_editor):
    PortfolioResult = apps.get_model('portfolio', 'PortfolioResult')
    for result in PortfolioResult.objects.all():
        strategies = result.strategies
        if not isinstance(strategies, dict):
            continue

        reverted = []
        for key, payload in strategies.items():
            entry = {'key': key}
            if isinstance(payload, dict):
                entry.update(payload)
            reverted.append(entry)

        result.strategies = reverted
        result.save(update_fields=['strategies'])


class Migration(migrations.Migration):

    dependencies = [
        ('portfolio', '0002_fix_strategies_default_dict'),
    ]

    operations = [
        migrations.RunPython(list_to_dict, dict_to_list),
    ]
