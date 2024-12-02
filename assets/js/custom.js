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
  

$(function(){
    // 이미지 클릭시 해당 이미지 모달
    $("span img").click(function(){
        let img = new Image();
        img.src = $(this).attr("src")
        $('.modalBox').html(img);
        $(".modal").show();
    });
    // 모달 클릭할때 이미지 닫음
    $(".modal").click(function (e) {
    	$(".modal").toggle();
    });
});