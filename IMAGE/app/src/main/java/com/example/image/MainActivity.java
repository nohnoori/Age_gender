package com.example.image;

import android.os.Environment;
import android.os.StrictMode;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.net.Socket;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    boolean mRun;
    private Socket clientSocket;
    private BufferedInputStream socketIn; //서버의 데이터 읽어 옴
    private PrintWriter socketOut; //서버에 데이터 전송
    private int port = 5503;
    private final String ip = "203.250.78.180";
    private DBOpenHelper mDbOpenHelper;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // StrictMode는 개발자가 실수하는 것을 감지하고 해결할 수 있도록 돕는 일종의 개발 툴
        // - 메인 스레드에서 디스크 접근, 네트워크 접근 등 비효율적 작업을 하려는 것을 감지하여
        //   프로그램이 부드럽게 작동하도록 돕고 빠른 응답을 갖도록 함, 즉  Android Not Responding 방지에 도움
        StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();
        StrictMode.setThreadPolicy(policy);

        try {
            clientSocket = new Socket(ip, port);

            socketIn = new BufferedInputStream(clientSocket.getInputStream());
            //DataInputStream dis = new DataInputStream(socketIn);
            socketOut = new PrintWriter(clientSocket.getOutputStream(), true);
            socketOut.println("ok");

            File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "MyDocApp");
            if (!file.exists()) {
                if (!file.mkdirs()) {
                    Log.d("MyDocApp", "failed to create directory");
                    return;
                }
            }
            File temp = new File(file.getPath()+"testfile.jpg");
            if(temp == null){

                return;
            }
            TextView text=(TextView)findViewById(R.id.hello);
            text.setText(temp.getAbsolutePath());

            FileOutputStream fos = new FileOutputStream(file);
            socketOut.println("됐음");

            int ch;
            while ((ch = socketIn.read()) != -1){
                fos.write(ch);
            }
//
//            mDbOpenHelper = new DBOpenHelper(this);
//            mDbOpenHelper.open();
//            mDbOpenHelper.create();

            socketOut.close();
            socketIn.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

//        Cursor iCursor = mDbOpenHelper.sortColumn();
//        Log.d("showDatabase", "DB Size: " + iCursor.getCount());
//
//        while (iCursor.moveToNext()) {
//            String tempIndex = iCursor.getString(iCursor.getColumnIndex("_id"));
//            byte[] imagedata = iCursor.getBlob(iCursor.getColumnIndex("image"));
//            Bitmap tempimage = BitmapFactory.decodeByteArray(imagedata, 0, imagedata.length);

//            //InputStream 의 값을 읽어와서 data 에 저장
//            String data = socketIn.readLine().trim();
//            //Message 객체를 생성, 핸들러에 정보를 보낼 땐 이 메세지 객체를 이용
//
//            byte[] bdata = data.getBytes();
//            Bitmap img = BitmapFactory.decodeByteArray(bdata, 0, bdata.length);
//            image.setImageBitmap(img);
//            mDbOpenHelper.insertColumn(data.getBytes());
        //}
    }
}

