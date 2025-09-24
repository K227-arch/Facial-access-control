document.addEventListener('DOMContentLoaded', () => {

    // Sidebar toggle functionality
    const sidebar = document.querySelector('.sidebar');
    const logo = document.querySelector('.logo');
    
    logo.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
        
        // Toggle visibility of nav text and logo text
        const navTexts = document.querySelectorAll('.nav-text');
        const logoText = document.querySelector('.logo-text');
        
        if (sidebar.classList.contains('collapsed')) {
            navTexts.forEach(text => text.style.display = 'none');
            logoText.style.display = 'none';
            sidebar.style.width = '60px';
        } else {
            navTexts.forEach(text => text.style.display = 'inline');
            logoText.style.display = 'inline';
            sidebar.style.width = '200px';
        }
    });
});