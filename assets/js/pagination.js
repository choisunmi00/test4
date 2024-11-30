document.addEventListener("DOMContentLoaded", function () {
    const list = document.getElementById("paginated-list"); // 전체 목록
    const listItems = list.querySelectorAll("li"); // 목록 항목들
    const prevButton = document.getElementById("prev-button");
    const nextButton = document.getElementById("next-button");
    const itemsPerPage = 5; // 페이지당 표시할 항목 수
    let currentPage = 1;
  
    const updatePagination = () => {
      // 전체 페이지 계산
      const totalPages = Math.ceil(listItems.length / itemsPerPage);
  
      // 목록 숨기기
      listItems.forEach((item, index) => {
        item.style.display =
          index >= (currentPage - 1) * itemsPerPage &&
          index < currentPage * itemsPerPage
            ? "list-item"
            : "none";
      });
  
      // 버튼 활성화/비활성화 설정
      prevButton.disabled = currentPage === 1;
      nextButton.disabled = currentPage === totalPages;
    };
  
    // 이전 버튼 클릭
    prevButton.addEventListener("click", () => {
      if (currentPage > 1) {
        currentPage--;
        updatePagination();
      }
    });
  
    // 다음 버튼 클릭
    nextButton.addEventListener("click", () => {
      if (currentPage * itemsPerPage < listItems.length) {
        currentPage++;
        updatePagination();
      }
    });
  
    // 초기 업데이트
    updatePagination();
  });
  