function openModal(image) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    modal.style.display = 'block';
    modalImage.src = image.src; // 클릭한 이미지의 소스를 가져옴
  }
  
  function closeModal() {
    const modal = document.getElementById('imageModal');
    modal.style.display = 'none'; // 모달 닫기
  }
  