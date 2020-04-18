function buttonAjax() {
    $.ajax({
        url:'/testAjax',
        type: 'post',
        success: function(response) {
            console.log(response);
        },
        error: function(error) {
            console.log(error);
        }
    });
};