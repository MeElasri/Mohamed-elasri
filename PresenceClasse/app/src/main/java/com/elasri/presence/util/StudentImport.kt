package com.elasri.presence.util

/** Turns any student-list file (.xlsx from Massar/Excel, or .csv) into classes. */
object StudentImport {

    /**
     * One [StudentTable.ParsedClass] per non-empty sheet (one for a CSV).
     * The class name comes from the file's title rows, else the sheet name, else the file name.
     */
    fun parseFile(bytes: ByteArray, fileName: String): List<StudentTable.ParsedClass> {
        val base = fileName.substringBeforeLast('.').ifBlank { "Classe" }
        return if (Xlsx.isXlsx(bytes)) {
            val sheets = Xlsx.read(bytes)
            sheets.mapNotNull { sh ->
                val fallback = if (sheets.size == 1 || sh.name.isBlank() || sh.name.startsWith("Sheet") ||
                    sh.name.startsWith("Feuil")
                ) base else sh.name
                StudentTable.extractClass(sh.rows, fallback).takeIf { it.students.isNotEmpty() }
            }
        } else {
            if (bytes.size > 8 && bytes[0] == 0xD0.toByte() && bytes[1] == 0xCF.toByte()) {
                throw Xlsx.NotXlsxException() // old .xls
            }
            val rows = Csv.rows(Csv.decode(bytes))
            listOf(StudentTable.extractClass(rows, base)).filter { it.students.isNotEmpty() }
        }
    }

    /** All students of a file, for importing into one existing class. */
    fun parseStudents(bytes: ByteArray, fileName: String): List<StudentTable.ParsedStudent> =
        parseFile(bytes, fileName).flatMap { it.students }
}
