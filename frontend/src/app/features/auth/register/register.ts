import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ReactiveFormsModule, FormBuilder, FormGroup, Validators, AbstractControl } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../../core/services/auth';

function passwordMatchValidator(control: AbstractControl) {
  const password = control.get('password')?.value;
  const confirm  = control.get('password_confirm')?.value;
  return password && confirm && password !== confirm ? { passwordMismatch: true } : null;
}

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterLink],
  templateUrl: './register.html',
  styleUrl: './register.scss',
})
export class Register {
  form: FormGroup;
  loading = false;
  errorMessages: string[] = [];

  constructor(
    private fb: FormBuilder,
    private authService: AuthService,
    private router: Router
  ) {
    this.form = this.fb.group(
      {
        email:            ['', [Validators.required, Validators.email]],
        username:         ['', [Validators.required, Validators.minLength(3)]],
        password:         ['', [Validators.required, Validators.minLength(8)]],
        password_confirm: ['', Validators.required],
      },
      { validators: passwordMatchValidator }
    );
  }

  onSubmit(): void {
    if (this.form.invalid || this.loading) return;
    this.loading = true;
    this.errorMessages = [];
    const { email, username, password } = this.form.value;

    this.authService.register(email, username, password).subscribe({
      next: () => this.router.navigate(['/dashboard']),
      error: (err) => {
        const errors = err.error;
        if (errors && typeof errors === 'object') {
          this.errorMessages = Object.entries(errors)
            .flatMap(([, msgs]) => (Array.isArray(msgs) ? msgs : [msgs])) as string[];
        } else {
          this.errorMessages = ['Registration failed. Please try again.'];
        }
        this.loading = false;
      },
    });
  }
}
