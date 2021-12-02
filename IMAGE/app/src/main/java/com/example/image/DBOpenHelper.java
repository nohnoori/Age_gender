package com.example.image;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.SQLException;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteDatabase.CursorFactory;
import android.database.sqlite.SQLiteOpenHelper;
import android.database.sqlite.SQLiteStatement;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import java.io.ByteArrayOutputStream;
import java.sql.Blob;

public class DBOpenHelper {

    private static final String DATABASE_NAME = "imageDatabase.db";
    private static final int DATABASE_VERSION = 1;
    public static SQLiteDatabase mDB;
    private DBHelper mDBHelper;
    private Context mCtx;

    private class DBHelper extends SQLiteOpenHelper{

        public DBHelper(Context context, String name, CursorFactory factory, int version) {
            super(context, name, factory, version);
        }

        @Override
        public void onCreate(SQLiteDatabase db){

            db.execSQL(Database.CreateDB._CREATE0);
        }
        @Override
        public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
            db.execSQL("DROP TABLE IF EXISTS "+Database.CreateDB._TABLENAME0);
            onCreate(db);
        }
    }
    public DBOpenHelper(Context context){
        this.mCtx = context;
    }

    public DBOpenHelper open() throws SQLException{
        mDBHelper = new DBHelper(mCtx, DATABASE_NAME, null, DATABASE_VERSION);
        mDB = mDBHelper.getWritableDatabase();
        return this;
    }

    public void create(){
        mDBHelper.onCreate(mDB);
    }

    public void close(){
        mDB.close();
    }

//    public void tempInsert(){
//        ContentValues values = new ContentValues();
//
//        byte[] data = ;
//        values.put(Database.CreateDB.IMAGE, data);
//    }

    //insertColumn
    public long insertColumn(byte[] image){
        SQLiteStatement p = mDB.compileStatement("INSERT INTO  imagetable values(?);");
        p.bindBlob(1, image);
        ContentValues values = new ContentValues();
        values.put(Database.CreateDB.IMAGE, image);

        return mDB.insert(Database.CreateDB._TABLENAME0, null, values);
    }
    // Delete All
    public void deleteAllColumns() {
        mDB.delete(Database.CreateDB._TABLENAME0, null, null);
    }

    // Delete DB
    public boolean deleteColumn(long id){
        return mDB.delete(Database.CreateDB._TABLENAME0, "_id="+id, null) > 0;
    }
    // Select DB
    public Cursor selectColumns(String[] id){
        Cursor c = mDB.rawQuery("SELECT * FROM imagetable WHERE _id = ?", id);
        return c;
    }
    // sort by column
    public Cursor sortColumn(){
        Cursor c = mDB.rawQuery( "SELECT * FROM imagetable ;", null);
        return c;
    }
    // convert from bitmap to byte array
    public static byte[] getBytes(Bitmap bitmap) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 0, stream);
        return stream.toByteArray();
    }

    // convert from byte array to bitmap
    public static Bitmap getImage(byte[] image) {
        return BitmapFactory.decodeByteArray(image, 0, image.length);
    }
}
