pub trait AlphaPlayer: Sized + Default {
    fn next_player(&self) -> Self;
    fn reward(&self, winner: Option<Self>) -> f32;
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum DefaultPlayer {
    Player1,
    Player2,
}

impl AlphaPlayer for DefaultPlayer {
    fn next_player(&self) -> DefaultPlayer {
        match self {
            DefaultPlayer::Player1 => DefaultPlayer::Player2,
            DefaultPlayer::Player2 => DefaultPlayer::Player1,
        }
    }

    fn reward(&self, winner: Option<DefaultPlayer>) -> f32 {
        match (self, winner) {
            (DefaultPlayer::Player1, None) => 0.0f32,
            (DefaultPlayer::Player1, Some(DefaultPlayer::Player1)) => 1.0f32,
            (DefaultPlayer::Player1, Some(DefaultPlayer::Player2)) => -1.0f32,
            (DefaultPlayer::Player2, None) => todo!(),
            (DefaultPlayer::Player2, Some(DefaultPlayer::Player1)) => -1.0f32,
            (DefaultPlayer::Player2, Some(DefaultPlayer::Player2)) => 1.0f32,
        }
    }
}

impl Default for DefaultPlayer {
    fn default() -> Self {
        DefaultPlayer::Player1
    }
}

impl From<&DefaultPlayer> for u8 {
    fn from(ob: &DefaultPlayer) -> u8 {
        match ob {
            DefaultPlayer::Player1 => 0,
            DefaultPlayer::Player2 => 1,
        }
    }
}
impl From<DefaultPlayer> for u8 {
    fn from(ob: DefaultPlayer) -> u8 {
        match ob {
            DefaultPlayer::Player1 => 0,
            DefaultPlayer::Player2 => 1,
        }
    }
}
