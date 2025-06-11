document.addEventListener('DOMContentLoaded', function() {
    // Menú móvil
    const navMenu = document.querySelector('.nav__menu');
    const navLinks = document.querySelector('.nav__link');

    navMenu.addEventListener('click', function() {
        navLinks.classList.toggle('nav__link--show');
    });

    // Cerrar menú móvil al hacer clic en un enlace
    document.querySelectorAll('.nav__links').forEach(link => {
        link.addEventListener('click', function() {
            navLinks.classList.remove('nav__link--show');
        });
    });

    // Funcionalidad del escáner
    window.startScan = function() {
        const preview = document.querySelector('.camera__preview');
        preview.innerHTML = '📸';
        preview.style.background = 'linear-gradient(45deg, #4CAF50, #66BB6A)';
        preview.style.color = 'white';

        setTimeout(() => {
            preview.innerHTML = '🍎';
            preview.style.background = '#f5f5f5';
            preview.style.color = '#666';
        }, 2000);
    };

    window.scanFruit = function() {
        const button = document.querySelector('.scan__button');
        const preview = document.querySelector('.camera__preview');

        button.textContent = 'Escaneando...';
        button.disabled = true;

        setTimeout(() => {
            preview.innerHTML = '✅';
            preview.style.background = 'linear-gradient(45deg, #4CAF50, #66BB6A)';
            preview.style.color = 'white';
            button.textContent = 'Fruta Identificada';

            setTimeout(() => {
                button.textContent = 'Escanear Fruta';
                button.disabled = false;
                preview.innerHTML = '📱';
                preview.style.background = '#f5f5f5';
                preview.style.color = '#666';
            }, 3000);
        }, 2000);
    };
});
