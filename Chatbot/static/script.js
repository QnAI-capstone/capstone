/* 메뉴버튼 */
const menuButton = document.getElementById('menuButton');
const closeButton = document.getElementById('closeMenuButton');
const dimmed = document.getElementById('dimmedBackground');
const sideMenu = document.getElementById('sideMenu');

menuButton.addEventListener('click', () => {
  dimmed.classList.remove('inactive');
  sideMenu.classList.remove('inactive');
  dimmed.classList.add('active');
  sideMenu.classList.add('active');
});

function closeMenu() {
  dimmed.classList.remove('active');
  sideMenu.classList.remove('active');
  dimmed.classList.add('inactive');
  sideMenu.classList.add('inactive');
}

dimmed.addEventListener('click', closeMenu);
closeButton.addEventListener('click', closeMenu);