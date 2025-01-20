from django import template

register = template.Library()


@register.filter
def with_index(value):
    return enumerate(value)
