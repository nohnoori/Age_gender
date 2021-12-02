package com.example.image;

import android.provider.BaseColumns;

public final class Database {
    public static final class CreateDB implements BaseColumns{
        public static final String IMAGE = "image";
        public static final String _TABLENAME0 = "imagetable";
        public static final String _CREATE0 = "create table if not exists "+_TABLENAME0+"("
                +_ID+" integer primary key autoincrement, "
                +IMAGE+" BLOB "+");";
    }
}
