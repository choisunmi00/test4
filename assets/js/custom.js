// function openModal(image) {
//     const modal = document.getElementById('imageModal');
//     const modalImage = document.getElementById('modalImage');
//     modal.style.display = 'block';
//     modalImage.src = image.src; // 클릭한 이미지의 소스를 가져옴
//   }
  
//   function closeModal() {
//     const modal = document.getElementById('imageModal');
//     modal.style.display = 'none'; // 모달 닫기
//   }
  

$(function () {
    // 이미지 클릭 시 모달 표시
    $(".author__avatar img").click(function () {
      let img = new Image();
      img.src = $(this).attr("src");
      img.style.maxWidth = "100%";
      img.style.maxHeight = "100%";
      $('.modalBox').html(img);
      $(".modal").fadeIn(); // 모달 표시
    });
  
    // 모달 클릭 시 닫기
    $(".modal").click(function () {
      $(this).fadeOut(); // 모달 닫기
    });
  
    // 모달 내부 이미지 클릭 시 닫히지 않도록 이벤트 전파 방지
    $(".modalBox").click(function (e) {
      e.stopPropagation();
    });
  });
  