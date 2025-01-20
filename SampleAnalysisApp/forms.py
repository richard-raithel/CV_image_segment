from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django import forms


class RegistrationForm(UserCreationForm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for visible in self.visible_fields():
            visible.field.widget.attrs['class'] = 'form-control'

    first_name = forms.CharField(max_length=50, required=False, help_text='Optional, please share your name.')
    last_name = forms.CharField(max_length=50, required=False, help_text='Optional, please share your name.')
    company_name = forms.CharField(max_length=250, required=True, help_text='Required, please share what company you work for.')
    email = forms.CharField(max_length=250, required=True, help_text='Required, please provide a valid email.')

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'company_name', 'email', 'password1', 'password2', )
