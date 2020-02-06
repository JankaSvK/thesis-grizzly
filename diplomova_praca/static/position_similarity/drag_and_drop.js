var available_id = 0;

function getAllImagesPositions() {
    let images = jQuery(".image");
    let images_information = [];
    for (var i = 0; i < images.length; i++) {
        let image = images.eq(i);
        images_information.push({
            url: image.children("img").attr("src"),
            top: image.position().top,
            left: image.position().left,
            width: image.width(),
            height: image.height()
        })
    }

    return images_information;
}

function sendRequest() {
    console.log("Sending request.");
    console.log(getAllImagesPositions());

    $.ajax({
        type: "POST",
        dataType: 'json',
        url: 'position_similarity/',
        data: {
            "json_data": JSON.stringify(getAllImagesPositions())
        },

    });
}

function reloadResizeable() {
    const image = jQuery(".image");

    image.draggable({
        containment: "#container",
        stop: function (event, ui) {
            // sendRequest()
        }
    });

    image.resizable({
        aspectRatio: true,
        containment: "#container",
        stop: function (event, ui) {
            // sendRequest()
        }
    });
}

(function ($) {
    $(document).ready(function () {
        $('#add_image').click(function () {
            const url = ($('#image_url').val());
            $("#container").append(`<div id="image_${available_id}" class="image ui-widget-content"><img src="${url}"></div>`);
            available_id++;

            reloadResizeable();
        });

        $('#send_request').click(function () {
            sendRequest();
        });
    });

})(jQuery);
