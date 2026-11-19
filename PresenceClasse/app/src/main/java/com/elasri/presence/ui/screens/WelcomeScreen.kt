package com.elasri.presence.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Login
import androidx.compose.material.icons.filled.Groups
import androidx.compose.material.icons.filled.HowToReg
import androidx.compose.material.icons.outlined.Description
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.elasri.presence.R

// Welcome screen palette (fixed, independent of the phone's dark mode, like the mock-up)
private val Navy = Color(0xFF13315C)
private val Teal = Color(0xFF16888A)
private val SoftBlue = Color(0xFFE7F0FA)
private val LineBlue = Color(0xFF8DB3DD)
private val Grey = Color(0xFF5F6B7A)

@Composable
fun WelcomeScreen(onStart: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .statusBarsPadding()
            .navigationBarsPadding()
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.height(28.dp))
        AttendanceLogo(Modifier.size(width = 112.dp, height = 100.dp))
        Spacer(Modifier.height(18.dp))
        Text(
            "Gestion des absences\ndes élèves",
            color = Navy,
            fontSize = 30.sp,
            lineHeight = 36.sp,
            fontWeight = FontWeight.ExtraBold,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 24.dp)
        )
        Spacer(Modifier.height(8.dp))
        Text(
            "Suivi de la présence en toute simplicité",
            color = Grey,
            fontSize = 17.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 24.dp)
        )
        Spacer(Modifier.height(12.dp))

        Image(
            painter = painterResource(R.drawable.accueil_illustration),
            contentDescription = "Un enseignant fait l'appel sur une tablette à l'entrée de la classe",
            contentScale = ContentScale.Crop,
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(756f / 560f)
        )

        Spacer(Modifier.height(20.dp))
        Row(
            modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.Top
        ) {
            Feature(Icons.Filled.HowToReg, "Saisie des\nabsences")
            WelcomeDivider()
            Feature(Icons.Filled.Groups, "Suivi de la\nprésence")
            WelcomeDivider()
            Feature(Icons.Outlined.Description, "Rapports des\nclasses")
        }

        Spacer(Modifier.height(24.dp))
        Button(
            onClick = onStart,
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(containerColor = Teal, contentColor = Color.White),
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp)
                .height(56.dp)
        ) {
            Text("Commencer", fontSize = 18.sp, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.width(10.dp))
            Icon(Icons.AutoMirrored.Filled.Login, contentDescription = null)
        }
        Spacer(Modifier.height(12.dp))
        Text(
            "Destiné aux cadres administratifs et pédagogiques",
            color = Grey,
            fontSize = 13.sp,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 24.dp)
        )
        Spacer(Modifier.height(20.dp))
    }
}

@Composable
private fun Feature(icon: ImageVector, label: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.width(100.dp)) {
        Box(
            modifier = Modifier.size(64.dp).background(SoftBlue, CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Icon(icon, contentDescription = null, tint = Navy, modifier = Modifier.size(32.dp))
        }
        Spacer(Modifier.height(8.dp))
        Text(
            label,
            color = Navy,
            fontSize = 14.sp,
            lineHeight = 18.sp,
            fontWeight = FontWeight.Medium,
            textAlign = TextAlign.Center
        )
    }
}

@Composable
private fun WelcomeDivider() {
    Box(
        Modifier
            .padding(top = 12.dp)
            .width(1.dp)
            .height(40.dp)
            .background(Color(0xFFDCE3EC))
    )
}

/** Clipboard with three pupils and a teal check badge (vector, sharp at any size). */
@Composable
fun AttendanceLogo(modifier: Modifier = Modifier) {
    Canvas(modifier) {
        val u = size.width / 112f // design grid: 112 x 100
        val stroke = 6f * u

        // clipboard
        drawRoundRect(
            color = Navy,
            topLeft = Offset(8f * u, 4f * u),
            size = Size(70f * u, 88f * u),
            cornerRadius = CornerRadius(8f * u),
            style = Stroke(width = stroke)
        )
        // three pupils + lines
        for (i in 0 until 3) {
            val cy = 22f * u + i * 24f * u
            drawCircle(Navy, radius = 5f * u, center = Offset(24f * u, cy - 4f * u))
            drawRoundRect(
                Navy,
                topLeft = Offset(17f * u, cy + 2f * u),
                size = Size(14f * u, 8f * u),
                cornerRadius = CornerRadius(4f * u)
            )
            val lineLen = if (i == 0) 30f else 22f
            drawLine(LineBlue, Offset(38f * u, cy - 3f * u), Offset((38f + lineLen) * u, cy - 3f * u), 4f * u, StrokeCap.Round)
            drawLine(LineBlue, Offset(38f * u, cy + 5f * u), Offset((38f + lineLen - 8f) * u, cy + 5f * u), 4f * u, StrokeCap.Round)
        }
        // teal badge with white border + check
        val c = Offset(84f * u, 70f * u)
        drawCircle(Color.White, radius = 25f * u, center = c)
        drawCircle(Teal, radius = 21f * u, center = c)
        val check = Path().apply {
            moveTo(c.x - 9f * u, c.y + 0.5f * u)
            lineTo(c.x - 2.5f * u, c.y + 7f * u)
            lineTo(c.x + 10f * u, c.y - 6f * u)
        }
        drawPath(check, Color.White, style = Stroke(width = 5f * u, cap = StrokeCap.Round, join = StrokeJoin.Round))
    }
}
