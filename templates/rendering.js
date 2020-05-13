$(document).ready(() => {

    $('#submit').on('click', () => {
		const sequence = $('#sequence').val();
		$.get('/', show);
	});
});

var i, tx, html, node;

node = document.getElementById("result-area");
tx = node.innerHTML;

html = "";
for (i = 0; i < tx.length; i++)
{
  html += "<span>" + tx.charAt(i) + "</span>";
}

node.innerHTML = html;

function show(){
    $('#result-area').innerHTML = "Predict btn is clicked!";
}
