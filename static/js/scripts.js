document.addEventListener('DOMContentLoaded', function() {
    // Automatically close alert messages after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeButton = new bootstrap.Alert(alert);
            closeButton.close();
        }, 5000);
    });

    // Password confirmation validation
    const passwordForm = document.querySelector('form[action*="register"], form[action*="profile"]');
    if (passwordForm) {
        const password = passwordForm.querySelector('input[name="password"], input[name="new_password"]');
        const confirmPassword = passwordForm.querySelector('input[name="confirm_password"]');
        
        if (password && confirmPassword) {
            confirmPassword.addEventListener('input', function() {
                if (password.value !== confirmPassword.value) {
                    confirmPassword.setCustomValidity("Passwords do not match");
                } else {
                    confirmPassword.setCustomValidity("");
                }
            });
            
            password.addEventListener('input', function() {
                if (confirmPassword.value && password.value !== confirmPassword.value) {
                    confirmPassword.setCustomValidity("Passwords do not match");
                } else {
                    confirmPassword.setCustomValidity("");
                }
            });
        }
    }
    
    // Enable all tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltipTriggerList.length > 0) {
        const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
    
    // Enable all popovers
    const popoverTriggerList = document.querySelectorAll('[data-bs-toggle="popover"]');
    if (popoverTriggerList.length > 0) {
        const popoverList = [...popoverTriggerList].map(popoverTriggerEl => new bootstrap.Popover(popoverTriggerEl));
    }
});