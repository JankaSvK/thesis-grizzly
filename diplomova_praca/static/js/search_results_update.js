function image_table(dom_element, images, submit_icon_path = null) {
    let ranking_table_content = [];
    for (const image of images) {
        const image_path = image['img_src'].replace(/\\/g, "/");
        // ranking_table_content.push("<div class='item'><img src=" + image['img_src'] + "></div>");
        ranking_table_content.push(`<div class='item wrap_corner_icon' style="width:320px; height:180px;"><img src="${image['img_src']}"/>
            <div class="top_right_icon_box">
                <a href="javascript:submit('${image_path}')"  class="icon">
                   <img src="${submit_icon_path}"/>
                </a>
            </div>
         </div>`);
    }

    dom_element.html(ranking_table_content.join(""));
}