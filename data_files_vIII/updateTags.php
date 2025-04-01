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
        $fileName = mysqli_real_escape_string($connection, $_POST["fileName"]);
        $tags = json_decode($_POST["tags"], true);

        //delete things so it can overwrite
        $deleteQuery = "DELETE FROM tags WHERE username = '$username' AND file_name = '$fileName'";
        if (!mysqli_query($connection, $deleteQuery)) 
        {
            echo json_encode(["success" => false, "message" => "Failed to delete existing tags"]);
            die();
        }
        // insert (new) data
        foreach ($tags as $tag) 
        {
            $tag = mysqli_real_escape_string($connection, $tag);
            $insertQuery = "INSERT INTO tags (file_name, username, tag) VALUES ('$fileName', '$username', '$tag')";
            if (!mysqli_query($connection, $insertQuery))
            {
                echo json_encode(["success" => false, "message" => "Failed to insert new tag: $tag"]);
                die();
            }
        }
    }


?>