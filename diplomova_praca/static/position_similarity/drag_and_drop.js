var available_id = 0;

function getAllImagesPositions() {
    let images = jQuery(".image");
    let canvas = jQuery("#container");
    let images_information = [];
    for (var i = 0; i < images.length; i++) {
        let image = images.eq(i);
        images_information.push({
            url: image.children("img").attr("src"),
            top: image.position().top / canvas.height(),
            left: image.position().left / canvas.width(),
            width: image.width() / canvas.width(),
            height: image.height() / canvas.height()
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
        url: "{% url 'position_similarity_post' %}",
        data: {
            "json_data": JSON.stringify(getAllImagesPositions())
        },
        success: [function (response) {
            console.log("Success", response);
            $('<p>Text</p>').appendTo('#msg');
        }],
        failure: [function (data) {
            console.log("Failure", data);
            $('<p>Text</p>').appendTo('#msg');
        }]
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
