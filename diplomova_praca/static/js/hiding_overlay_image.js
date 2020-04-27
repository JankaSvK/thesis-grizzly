var num_hints_for_overlay = 0;
var overlay_timeout;

function showImageTemporarly(to_display) {
    num_hints_for_overlay++;
    clearTimeout(overlay_timeout);

    $("#show_image_show").toggle();
    $("#show_image_hide").toggle();

    const overlay_image = $("#overlay");
    to_display ? overlay_image.show() : overlay_image.hide();
    overlay_image.css('z-index', (to_display ? 1 : -1));

    if (to_display) { // If showed
        overlay_timeout = setTimeout(function () {
            showImageTemporarly(false)
        }, 3000); // Hide it after 3 seconds
    }
}