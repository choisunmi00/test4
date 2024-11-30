document.addEventListener("DOMContentLoaded", function () {
    const list = document.getElementById("paginated-list"); // 전체 목록
    const listItems = Array.from(list.querySelectorAll("li")); // 목록 항목 배열화
    const prevButton = document.getElementById("prev-button");
    const nextButton = document.getElementById("next-button");
    const itemsPerPage = 5; // 페이지당 표시할 항목 수
    let currentPage = 1; // 현재 페이지
  
    // 페이지네이션 업데이트 함수
    const updatePagination = () => {
      const totalPages = Math.ceil(listItems.length / itemsPerPage);
  
      // 항목 표시/숨기기
      listItems.forEach((item, index) => {
        if (index >= (currentPage - 1) * itemsPerPage && index < currentPage * itemsPerPage) {
          item.style.display = "list-item"; // 표시
        } else {
          item.style.display = "none"; // 숨기기
        }
      });
  
      // 이전/다음 버튼 활성화/비활성화
      prevButton.disabled = currentPage === 1;
      nextButton.disabled = currentPage === totalPages;
    };
  
    // 이전 버튼 클릭 이벤트
    prevButton.addEventListener("click", () => {
      if (currentPage > 1) {
        currentPage--;
        updatePagination();
      }
    });
  
    // 다음 버튼 클릭 이벤트
    nextButton.addEventListener("click", () => {
      if (currentPage * itemsPerPage < listItems.length) {
        currentPage++;
        updatePagination();
      }
    });
  
    // 초기 업데이트
    updatePagination();
  });
  