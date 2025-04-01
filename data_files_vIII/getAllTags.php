<?php
    header("Access-Control-Allow-Origin: http://localhost:3000");
    header("Access-Control-Allow-Headers: Content-Type");
    header("Content-Type: application/json; charset=UTF-8");
    session_start();

    $connection = mysqli_connect("localhost", "root", "", "webologyTask");

    if (mysqli_connect_error()) {
        echo json_encode(["success" => false, "message" => mysqli_connect_error()]);
        die();
    }

    if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST["username"])) 
    {
        $username = mysqli_real_escape_string($connection, $_POST["username"]);
        $fileName = mysqli_real_escape_string($connection, $_POST["fileName"]);
        
        $query = "SELECT `tag` FROM `tags` WHERE `username` = '$username'";
        $result = mysqli_query($connection, $query);
        
        if (mysqli_num_rows($result) > 0) 
        {
            $tags = [];
            while ($row = mysqli_fetch_assoc($result)) 
            {
                $tags[] = $row["tag"];
            }
            echo json_encode(["success" => true, "tags" => $tags]);
        } else {
            echo json_encode(["success" => false, "message" => "Error retrieving tags"]);
        }
    } else {
        echo json_encode(["success" => false, "message" => "Invalid request"]);
    }
?>
