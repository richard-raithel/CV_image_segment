// // Number of items to show per page
// var itemsPerPage = 4;
//
// // Current page
// var currentPage = 1;

// Get the item container and pagination buttons
var itemContainer = $('#card-deck');
// var prevBtn = $('#prev-btn');
// var nextBtn = $('#next-btn');
//
// // Calculate the total number of pages
// var totalItems = itemContainer.children().length;
// var totalPages = Math.ceil(totalItems / itemsPerPage);
//
// // Function to show items for the current page
// function showPage(page) {
//     itemContainer.children().hide();
//     itemContainer.children().slice((page - 1) * itemsPerPage, page * itemsPerPage).show();
// }
//
// // Show the first page initially
// showPage(currentPage);
//
// prevBtn.on('click', function () {
//     if (currentPage > 1) {
//         currentPage--;
//         showPage(currentPage);
//     }
// });
//
// nextBtn.on('click', function () {
//     if (currentPage < totalPages) {
//         currentPage++;
//         showPage(currentPage);
//     }
// });

$('#search').on('submit', function (event) {
    event.preventDefault();
    var formulation = $('#formulation').val().trim();
    var batch = $('#batch').val().trim();
    const loadMoreBtn = document.getElementById("load-more-btn");

    // If both search terms are empty, show all items and add class "card-export" if not present
    if (formulation === '' && batch === '') {
        itemContainer.children().show().addClass('card-export').removeClass('card-class');
        itemContainer.children(':not(.card-export)').addClass('card-export');
        loadMoreBtn.style.display = "block";
        displayCards();
    } else {
        // If at least one search term is not empty, hide all items and remove class "card-export"
        itemContainer.children().hide().removeClass('card-export');

        // Filter the items based on the search terms
        itemContainer.children().filter(function() {
            var itemFormulation = $(this).attr('data-image-name').toLowerCase();
            var itemBatch = $(this).attr('data-batch').toLowerCase();

            var matchFormulation = formulation === '' || itemFormulation.includes(formulation.toLowerCase());
            var matchBatch = batch === '' || itemBatch.includes(batch.toLowerCase());

            return matchFormulation && matchBatch;
        }).show().addClass('card-export').removeClass('card-class');

        loadMoreBtn.style.display = "none";
    }
});


// // Update pagination buttons
// function updatePagination() {
//     // Calculate total number of items after filtering
//     var totalItems = itemContainer.children(':visible').length;
//     // Calculate total number of pages
//     var totalPages = Math.ceil(totalItems / itemsPerPage);
//     // Disable/enable previous button based on current page
//     if (currentPage === 1) {
//         prevBtn.prop('disabled', true);
//     } else {
//         prevBtn.prop('disabled', false);
//     }
//     // Disable/enable next button based on current page
//     if (currentPage === totalPages) {
//         nextBtn.prop('disabled', true);
//     } else {
//         nextBtn.prop('disabled', false);
//     }
// }
//
// // Update pagination buttons on item filtering
// itemContainer.on('show.bs.collapse', function () {
//     updatePagination();
// });

$(document).ready(function () {
    $('.btn-export').click(function () {
        var data = [];

        // Loop through each card with class card-export and extract the data
        $('.card-export').each(function () {
            var name = $(this).find('.image_name').text();
            var user_rating = $(this).find('.user_rating').text();
            var percent = $(this).find('.percent').text();
            var model_rating = $(this).find('.model_rating').text();

            // Add the data to the list
            data.push({
                'name': name,
                'user_rating': user_rating,
                'percent': percent,
                'model_rating': model_rating
            });
        });

        // Convert the data to JSON format
        var json_data = JSON.stringify(data);
        const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;

        // Make the AJAX request
        $.ajax({
            headers: {'X-CSRFToken': csrftoken},
            type: 'POST',
            url: $(this).data('url'),
            data: json_data,
            success: function(data) {
                // Create a hidden <a> element with the URL of the generated file
                var downloadLink = document.createElement('a');
                downloadLink.href = window.URL.createObjectURL(new Blob([data]));

                // Set the download attribute of the <a> element to the desired filename
                downloadLink.setAttribute('download', 'export.csv');

                // Add the <a> element to the DOM and programmatically click on it to trigger the download
                document.body.appendChild(downloadLink);
                downloadLink.click();

                // Clean up the <a> element
                document.body.removeChild(downloadLink);

                console.log("Export successful!");
            },
            error: function (xhr, textStatus, errorThrown) {
                console.log("Export failed: " + errorThrown);
            }
        });
    });
});

$(document).ready(function () {
    $("#create-report-btn").on("click", function () {
        var data = [];
        const customerName = $("#customer-name").val().trim();
        const comments = $("#comments-text").val().trim();

        // Loop through each card with class card-export and extract the data
        $('.card-export').each(function () {
            var $card = $(this);

            var name = $card.find('.image_name').text();
            var user_rating = $card.find('.user_rating').text();
            var percent = $card.find('.percent').text();
            var model_rating = $card.find('.model_rating').text();

            // Get the base64 strings directly from the src attributes
            var imageBase64 = $card.find('.card-img-top').attr('src');
            var maskBase64 = $card.find('.mask-image').attr('src');

            // Add the data to the list, including the image and mask base64 strings
            data.push({
                'name': name,
                'user_rating': user_rating,
                'percent': percent,
                'model_rating': model_rating,
                'image': imageBase64,
                'mask': maskBase64,
            });
        });

        // Convert the data to JSON format
        var json_data = JSON.stringify({
            customerName: customerName,
            comments: comments,
            data: data
        });
        const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;

        fetch($(this).data('url'), {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrftoken,
                'Content-Type': 'application/json'
            },
            body: json_data
        })
        .then(response => response.blob())
        .then(data => {
            // Create a hidden <a> element with the URL of the generated file
            var downloadLink = document.createElement('a');
            downloadLink.href = window.URL.createObjectURL(data);

            // Set the download attribute of the <a> element to the desired filename
            downloadLink.setAttribute('download', 'report.pdf');

            // Add the <a> element to the DOM and programmatically click on it to trigger the download
            document.body.appendChild(downloadLink);
            downloadLink.click();

            // Clean up the <a> element
            document.body.removeChild(downloadLink);

            console.log("Export successful!");
        })
        .catch(error => {
            console.log("Export failed: " + error);
        });

        // Close the modal
        $("#createReportModal").modal("hide");

    });
});

// function displayCards() {
//   const cardDeck = document.getElementById("card-deck");
//   const cards = cardDeck.children;
//   const numToShow = 4
//
//   // hide all cards after the 4th one
//   for (let i = numToShow; i < cards.length; i++) {
//     cards[i].style.display = "none";
//   }
//
//   // add click event listener to load more button
//   const loadMoreBtn = document.getElementById("load-more-btn");
//     // if there are less or equal to numToShow cards, hide the load more button
//   if (cards.length <= numToShow) {
//     loadMoreBtn.style.display = "none";
//     return; // no need to add event listener, so exit the function
//   }
//   let nextIndex = numToShow; // starting index of next set of cards to display
//   loadMoreBtn.addEventListener("click", function() {
//     // display next set of 4 cards
//     for (let i = nextIndex; i < nextIndex + 4 && i < cards.length; i++) {
//       cards[i].style.display = "block";
//     }
//     nextIndex += numToShow;
//
//     // hide load more button if all cards have been displayed
//     if (nextIndex >= cards.length) {
//       loadMoreBtn.style.display = "none";
//     }
//   });
// }

function displayCards() {
  const cardDeck = document.getElementById("card-deck");
  const cards = cardDeck.children;
  const numToShow = 4;

  // hide all cards after the 4th one
  for (let i = numToShow; i < cards.length; i++) {
    cards[i].style.display = "none";
  }

  // add click event listener to load more button
  const loadMoreBtn = document.getElementById("load-more-btn");

  // if there are less or equal to numToShow cards, hide the load more button
  if (cards.length <= numToShow) {
    loadMoreBtn.style.display = "none";
    return; // no need to add event listener, so exit the function
  }

  let nextIndex = numToShow; // starting index of next set of cards to display
  loadMoreBtn.addEventListener("click", function() {
    // display next set of 4 cards
    for (let i = nextIndex; i < nextIndex + 4 && i < cards.length; i++) {
      cards[i].style.display = "block";
    }
    nextIndex += numToShow;

    // hide load more button if all cards have been displayed
    if (nextIndex >= cards.length) {
      loadMoreBtn.style.display = "none";
    }
  });
}


window.onload = function() {
  displayCards();
};
