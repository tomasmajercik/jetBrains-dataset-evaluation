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

    if ($_SERVER["REQUEST_METHOD"] == "POST") 
    {
        $username = mysqli_real_escape_string($connection, $_POST["username"]);
        $tag = mysqli_real_escape_string($connection, $_POST["tag"]);

        $fileQuery = "SELECT file_name FROM `tags` WHERE `tag` = '$tag'";
        $filesResult = mysqli_query($connection, $fileQuery);

        if ($filesResult) 
        {
            $files = [];
            while ($row = mysqli_fetch_assoc($filesResult)) 
            {
                $files[] = $row["file_name"];
            }
            echo json_encode(["success" => true, "files" => $files]);
        } 
        else 
        {
            echo json_encode(["success" => false, "message" => "Error retrieving tags"]);
        }
    } 
    else 
    {
        echo json_encode(["success" => false, "message" => "Invalid request"]);
    }

    mysqli_close($connection);
?>
