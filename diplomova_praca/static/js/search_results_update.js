function image_table(dom_element, images) {
    let ranking_table_content = [];
    for (const image of images) {
        ranking_table_content.push("<div class='item'><img src=" + image['img_src'] + "></div>");
    }

    dom_element.html(ranking_table_content.join(""));
}
