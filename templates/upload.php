<?php
$target_dir = "uploads/";
$target_file = $target_dir . basename($_FILES["fileToUpload"]["name"]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));

// Check if image file is a actual image or fake image
if(isset($_POST["submit"])) {
  $check = getimagesize($_FILES["fileToUpload"]["tmp_name"]);
  if($check == false) {
    
    $uploadOk = 1;
  } else {

    $uploadOk = 0;
  }
}

// Check if file already exists
if (file_exists($target_file)) {

  $uploadOk = 0;
}


// Allow certain file formats
if($imageFileType != "fasta") {
  $uploadOk = 0;
}

// Check if $uploadOk is set to 0 by an error
if ($uploadOk == 0) {

// if everything is ok, try to upload file
} else {
  if (move_uploaded_file($_FILES["fileToUpload"]["tmp_name"], $target_file)) {

  } else {

  }
}
?>
<!DOCTYPE html>
<html lang="en" xmlns:th="http://www.w3.org/1999/xhtml">
<head>
    <meta charset="UTF-8">
    <meta content="width=device-width, initial-scale=1" name="viewport">
    <link rel='icon' href="static/transparent.png" type='image/x-icon'/>
    <title>VPRE</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
    <link rel="stylesheet" href="static/main.css">
    <link href="https://fonts.googleapis.com/css2?family=Lato:wght@300&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans&family=Raleway:wght@500&display=swap" rel="stylesheet">
</head>





<script type="text/javascript">
    function show(){
    document.getElementById("result-area").innerHTML= "Predicted sequences for SARS-CoV-2 from file";

    tx = "Predicted sequences for SARS-CoV-2 from file";
    html = "";
    for (i = 0; i < tx.length; i++)
    {
      html += "<span>" + tx.charAt(i) + "</span>";
    }
    document.getElementById("result-area").innerHTML = html;
    }

    function visualize(){
    tx = document.getElementById("sequence").value;

    html = "";
    for (i = 0; i < tx.length; i++)
    {
      html += "<span>" + tx.charAt(i) + "</span>";
    }
    document.getElementById("result-area").innerHTML = html;
    }
</script>


<body>

<header>
<div class="menu">
        <nav>
               <ul>
                    <li><a href="#">Tool</a></li>
                    <li><img src="static/virosight-logo.png"></li>
                    <li><a href="https://ubcvirosight.godaddysites.com/">Home</a></li>
               </ul>
        </nav>
</div>
    <br>
    <h1><b>Viral Predictor for mRNA Evolution (VPRE)</b></h1>
    <h2><b>A software tool brought to you by UBC Virosight</b></h2>
    <br>
</header>

   <div class="content">
   <div class="intro">
   <p>VPRE uses a viral genome and species name inputted by the user to predict the  most probable mutated sequences based on the species' phylogeny. Our algorithm uses deep learning to produce this output, implementing both a neural network training set and Markov models.</p>
   </div>

<div class="container" style="display: flex">
    <form class="form-group" action="upload.php" enctype="multipart/form-data" method="post" style="width: 50%; margin-top: 2%">
      <div class="preview"></div>
        <label><h3><b>Viral Genome inputs:</b></h3></label>
        <p><img src="static/1.png">Input the species name of your virus below:</p>
        <textarea class="form-control" rows="16" id="sequence" placeholder="Species name..."></textarea>

        <p><img src="static/2.png">Input your viral genome below:</p>
        <input type="file" name="fileToUpload" id="fileToUpload" hidden="hidden"/>
    <button type="button" class="btn btn-outline-success" id="custom-button">Choose a file</button>
    <span id="custom-text">No file chosen</span>

       <script type="text/javascript">
            const realFileBtn = document.getElementById("fileToUpload");
            const customBtn = document.getElementById("custom-button");
            const customTxt = document.getElementById("custom-text");
            customBtn.addEventListener("click", function() {
                realFileBtn.click();

            });

           realFileBtn.addEventListener("change",function() {
              if (realFileBtn.value) {
                  customTxt.innerHTML = realFileBtn.value.match(/[\/\\]([\w\d\s\.\-\(\)]+)$/)[1];
              } else {
                  customTxt.innerHTML = "No file chosen"
              }
           });
        </script>

        <p><img src="static/3.png">Run your inputs through VPRE:</p>
        <input type="submit" id="tool" class="btn btn-outline-success" style="margin-top:2%" onclick="show()"></button>
        <button type="submit" id="tool" class="btn btn-outline-success" style="margin-top:2%; margin-left: 2%" onclick="visualize()">Visualize</button>
    </form>
    <div class="form-group" style="width:50%; margin-left: 5%; margin-top: 2%;">
        <label><h4><b>Predicted mRNA sequences:</b></h4></label>
        <div class="visualization-area">
            <p id="result-area"></p>
        </div>
        <a href="/static/prediction/PredictedSequence.fasta" download><button type="submit" id="download" class="btn btn-outline-primary" style="margin-top:2%px; margin-left: 210px;" onclick="visualize()">Download</button></a>
    </div>
    </div>
    </div>

    <script type="text/javascript">
        $(document).ready(function() {
            $(".btn-outline-success").click(function(){
                $(".form-group").ajaxForm({target: '.preview'}).submit();
            });
        });
    </script>
</body>
</html>
