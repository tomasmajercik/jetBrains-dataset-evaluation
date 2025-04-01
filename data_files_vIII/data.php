<?php
    header("Access-Control-Allow-Origin: http://localhost:3000");
    header("Content-Type: application/json; charset=UTF-8");
    session_start();
    
    $connection = mysqli_connect("localhost", "root", "", "webologyTask"); 
    
    if(mysqli_connect_error())
    {
        echo mysqli_connect_error();
        die();
    }
        
    function generateToken() // Simple token generation
    {
        return bin2hex(random_bytes(16)); 
    }

    //chenk if the post was submitted
    if ($_SERVER["REQUEST_METHOD"] == "POST") 
    {
        $name = mysqli_real_escape_string($connection, $_POST["name"]);
        $password = mysqli_real_escape_string($connection, $_POST["password"]);

        $query = "SELECT * FROM `users` WHERE `username` = '$name' AND `password` = '$password'";
        $result = mysqli_query($connection, $query);
        $token = generateToken();

        if (mysqli_num_rows($result) > 0) 
        {
            $query = "UPDATE `users` SET `token` = '$token' WHERE `username` = '$name' AND `password` = '$password'";
            mysqli_query($connection, $query);
            echo json_encode(["success" => true, "token" => $token]);
        } 
        else 
        {
            echo json_encode(["success" => false]);
        }
    }
    else
    {
        echo json_encode(["success" => false]);
    }



?>
