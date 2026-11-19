package com.elasri.presence.util

import java.time.Instant
import java.time.LocalDate
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.Locale

private val longFmt = DateTimeFormatter.ofPattern("EEEE d MMMM yyyy", Locale.FRENCH)
private val shortFmt = DateTimeFormatter.ofPattern("dd/MM/yyyy", Locale.FRENCH)

fun LocalDate.fmtLong(): String =
    longFmt.format(this).replaceFirstChar { it.titlecase(Locale.FRENCH) }

fun LocalDate.fmtShort(): String = shortFmt.format(this)

/** Material DatePicker works with UTC midnight millis. */
fun LocalDate.toUtcMillis(): Long = atStartOfDay(ZoneOffset.UTC).toInstant().toEpochMilli()

fun Long.utcMillisToLocalDate(): LocalDate =
    Instant.ofEpochMilli(this).atZone(ZoneOffset.UTC).toLocalDate()

/** Moroccan school year starts in September. */
fun schoolYearStart(today: LocalDate = LocalDate.now()): LocalDate =
    LocalDate.of(if (today.monthValue >= 9) today.year else today.year - 1, 9, 1)

fun schoolYearLabel(today: LocalDate = LocalDate.now()): String {
    val y = schoolYearStart(today).year
    return "$y-${y + 1}"
}
