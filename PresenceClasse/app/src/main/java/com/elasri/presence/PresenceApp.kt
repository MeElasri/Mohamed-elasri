package com.elasri.presence

import android.app.Application
import com.elasri.presence.data.FirebaseRepo

class PresenceApp : Application() {
    val repository: FirebaseRepo by lazy { FirebaseRepo(this) }
}
