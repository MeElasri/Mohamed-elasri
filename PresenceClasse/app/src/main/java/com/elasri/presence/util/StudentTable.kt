package com.elasri.presence.util

/**
 * Finds the student list inside a table of cells (from CSV or an Excel sheet).
 * Works with Massar exports (Arabic or French headers, title rows above the table)
 * and with simple hand-made lists.
 */
object StudentTable {

    data class ParsedStudent(val name: String, val code: String)
    data class ParsedClass(val name: String, val students: List<ParsedStudent>)

    /** Massar codes: one letter + digits, e.g. G123456789. */
    val massarRegex = Regex("^[A-Za-z][0-9]{6,12}$")

    /** Lowercase, unify Arabic alef/hamza forms, remove tatweel and extra spaces. */
    fun norm(s: String): String = s.lowercase()
        .replace('إ', 'ا').replace('أ', 'ا').replace('آ', 'ا')
        .replace("ـ", "")
        .replace(Regex("\\s+"), " ")
        .trim()

    private fun isCodeHeader(h: String): Boolean =
        "massar" in h || h == "code" || "code eleve" in h || "code élève" in h || h == "cne" ||
            "رقم التلميذ" in h || "رمز التلميذ" in h || "رمز مسار" in h || h == "الرمز" || "رقم مسار" in h

    private fun isFullNameHeader(h: String): Boolean =
        "nom et prénom" in h || "nom et prenom" in h || "nom complet" in h || "nom & prénom" in h ||
            ("الاسم" in h && "النسب" in h) || "الاسم الكامل" in h

    private fun isLastNameHeader(h: String): Boolean =
        !isFullNameHeader(h) && (h == "nom" || h.startsWith("nom ") || h == "name" || h == "النسب" ||
            h.startsWith("النسب") || "nom de famille" in h)

    private fun isFirstNameHeader(h: String): Boolean =
        !isFullNameHeader(h) && ("prénom" in h || "prenom" in h || h == "الاسم" || h.startsWith("الاسم "))

    private fun isHeaderRow(r: List<String>): Boolean {
        val h = r.map(::norm)
        val hasName = h.any { isFullNameHeader(it) || isLastNameHeader(it) || isFirstNameHeader(it) }
        return hasName || h.any(::isCodeHeader)
    }

    fun extractStudents(rows: List<List<String>>): List<ParsedStudent> {
        val clean = rows.map { r -> r.map { it.trim() } }.filter { r -> r.any { it.isNotBlank() } }
        if (clean.isEmpty()) return emptyList()

        // 1) Table with a header row (Massar files have several title rows first).
        val headerIdx = clean.take(25).indexOfFirst(::isHeaderRow)
        if (headerIdx >= 0) {
            val h = clean[headerIdx].map(::norm)
            val iCode = h.indexOfFirst(::isCodeHeader)
            val iFull = h.indexOfFirst(::isFullNameHeader)
            val iLast = h.indexOfFirst(::isLastNameHeader)
            val iFirst = h.indexOfFirst(::isFirstNameHeader)
            val parsed = clean.drop(headerIdx + 1).mapNotNull { r ->
                fun cell(i: Int) = if (i >= 0) r.getOrNull(i).orEmpty() else ""
                val code = cell(iCode)
                val name = when {
                    iFull >= 0 && cell(iFull).isNotBlank() -> cell(iFull)
                    else -> listOf(cell(iLast), cell(iFirst)).filter { it.isNotBlank() }.joinToString(" ")
                }
                // Stop at footer lines ("Total", signatures…): a student row needs a name,
                // and when the file has a code column, a valid code.
                when {
                    name.isBlank() -> null
                    iCode >= 0 && !massarRegex.matches(code) -> null
                    else -> ParsedStudent(name, code)
                }
            }
            if (parsed.isNotEmpty()) return parsed
        }

        // 2) No recognised header: rows containing a Massar code, or plain name lists.
        val withCode = clean.filter { r -> r.any { massarRegex.matches(it) } }
        val source = if (withCode.size >= 2) withCode else clean
        return source.mapNotNull { r ->
            val cells = r.filter { it.isNotBlank() }
            val code = cells.firstOrNull { massarRegex.matches(it) }.orEmpty()
            val name = cells.filter { c ->
                c != code && !c.all { it.isDigit() || it == '/' || it == '-' || it == '.' }
            }.take(2).joinToString(" ")
            if (name.isBlank()) null else ParsedStudent(name, code)
        }
    }

    /**
     * Looks in the title rows for the class name, e.g. "Classe : TCS-3", "القسم : 1BACSEF-2",
     * or a "Classe" cell followed by the value in the next cell.
     */
    fun findClassName(rows: List<List<String>>): String? {
        val keys = listOf("classe", "القسم", "الفوج", "groupe", "division")
        for (r in rows.take(15)) {
            val cells = r.map { it.trim() }
            for ((i, c) in cells.withIndex()) {
                val n = norm(c)
                if (keys.none { n.startsWith(it) }) continue
                // "Classe : TCS-3" in a single cell
                val after = c.substringAfter(':', "").substringAfter('：', "").trim()
                if (after.isNotBlank()) return after
                // value in a following cell
                cells.drop(i + 1).firstOrNull { it.isNotBlank() && it != ":" }?.let {
                    return it.removePrefix(":").trim()
                }
            }
        }
        return null
    }

    fun extractClass(rows: List<List<String>>, fallbackName: String): ParsedClass =
        ParsedClass(findClassName(rows) ?: fallbackName, extractStudents(rows))
}
