/* 메뉴버튼 */
const menuButton = document.getElementById('menuButton');
const closeButton = document.getElementById('closeMenuButton');
const dimmed = document.getElementById('dimmedBackground');
const subMenu = document.getElementById('sideMenu');

menuButton.addEventListener('click', () => {
  dimmed.classList.add('active');
  subMenu.classList.add('active');
});

function closeMenu() {
  dimmed.classList.remove('active');
  subMenu.classList.remove('active');
}

dimmed.addEventListener('click', closeMenu);
closeButton.addEventListener('click', closeMenu);