$(document).ready(function(){
    var fileTarget = $('.upload'); 
    fileTarget.on('change', function(){
        if(window.FileReader){ 
            var filename = $(this)[0].files[0].name; 
        } else {
            var filename = $(this).val().split('/').pop().split('\\').pop(); // 파일명
        } 
        $(this).siblings('.filename').val(filename); 
    }); 
});