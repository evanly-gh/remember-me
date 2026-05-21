/**
 * ChoppedScoreDisplay — the fun, flashy chopped-score card.
 *
 * Drop into EditProfileScreen wherever you want the score to appear,
 * gated by the Settings toggle. Animates a counter from 0 → final
 * score on mount, picks tier colours / labels / captions, springs into
 * view, and pulses for the worst tier.
 *
 * All animations use the native driver where possible. The counter
 * itself uses a listener (Text can't read from Animated.Value
 * directly), which forces useNativeDriver: false for that timing —
 * acceptable since it's a one-off 1.5 s tween.
 *
 * Self-contained: doesn't depend on the parent's theme tokens, only
 * on the boolean `isDark` flag for the card background.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Animated,
  Easing,
} from 'react-native';

// ── Tier definitions ──────────────────────────────────────────────
// Ranges are contiguous: a score lands in the first tier whose
// `max` it falls below.  Gigachad covers <20 because the user's
// original spec left 10–20 ambiguous; folding it into Gigachad is
// the kindest read.
const TIERS = [
  {
    max: 20,
    label: 'Gigachad',
    emoji: '🗿',
    color: '#00C853',   // vivid neon green
    glow: '#69F0AE',
    captions: [
      'Built different.',
      'Sigma grindset confirmed.',
      'The genetic lottery winner.',
      'Built like the Greeks intended.',
      'Approach with caution: aura levels critical.',
    ],
  },
  {
    max: 35,
    label: 'Lookin good!',
    emoji: '✨',
    color: '#66BB6A',   // light green
    glow: '#A5D6A7',
    captions: [
      "I bet this is just an everyday thing for you.",
      'Effortless aura unlocked.',
      'Pretty privilege detected.',
      'Naturally photogenic.',
      'The mirror is friendly today.',
    ],
  },
  {
    max: 65,
    label: 'Average Joe',
    emoji: '🟨',
    color: '#D4A017',   // mustard yellow
    glow: '#FFD54F',
    captions: [
      'Sucks to be normal, but oh well.',
      'Solidly mid.',
      'The median has entered the chat.',
      'Statistically unremarkable.',
      'Vibes: distinctly room-temperature.',
    ],
  },
  {
    max: 80,
    label: 'Kinda Chopped',
    emoji: '😬',
    color: '#EF5350',   // light red / coral
    glow: '#FFCDD2',
    captions: [
      'Careful there.',
      'My condolences.',
      'Skincare? Have you tried it?',
      'Could be worse. Could also be much better.',
      'The lighting is conspiring against you.',
    ],
  },
  {
    max: 95,
    label: 'Chopped',
    emoji: '🪓',
    color: '#E53935',   // bright red
    glow: '#EF9A9A',
    captions: [
      'Yikes.',
      'Recovery is possible.',
      'Have you tried smiling?',
      'Tough scene.',
      'Lock in, king/queen.',
    ],
  },
  {
    max: 101,                 // catches 95–100 inclusive
    label: 'Megachopped',
    emoji: '💀',
    color: '#FF1744',   // deep alarming red
    glow: '#FF8A80',
    captions: [
      'Catastrophic.',
      'Have you considered a paper bag?',
      'Maybe try a filter.',
      'The model wept.',
      'New PR (personal regression).',
    ],
  },
];

function getTier(score) {
  return TIERS.find((t) => score < t.max) || TIERS[TIERS.length - 1];
}

// Pick one caption per mount so the page doesn't re-roll on every
// re-render but does feel fresh between visits.
function useRandomCaption(tier) {
  return useMemo(
    () => tier.captions[Math.floor(Math.random() * tier.captions.length)],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tier.label]
  );
}

export default function ChoppedScoreDisplay({ score, isDark = false }) {
  // The 0 → score counter. We use a non-native-driver tween because
  // Text can't read from an Animated.Value directly; the listener
  // updates plain React state each frame.
  const counterValue = useRef(new Animated.Value(0)).current;
  const [displayed, setDisplayed] = useState(0);

  // Entrance: scale from 0.85 → 1.0 with a soft spring.
  const entranceScale = useRef(new Animated.Value(0.85)).current;
  const entranceOpacity = useRef(new Animated.Value(0)).current;

  // Continuous pulse for the worst tier so it visibly throbs.
  const pulseScale = useRef(new Animated.Value(1)).current;

  const tier = getTier(score);
  const caption = useRandomCaption(tier);
  const isMegaChopped = tier.label === 'Megachopped';

  useEffect(() => {
    // 1. Snap counter back to 0 in case the screen is re-mounted with
    //    a different score.
    counterValue.setValue(0);
    setDisplayed(0);

    const listener = counterValue.addListener(({ value }) => {
      setDisplayed(Math.round(value));
    });

    // 2. Count up to the real score over 1.5 s with an ease-out so it
    //    decelerates as it hits the final number.
    Animated.timing(counterValue, {
      toValue: score,
      duration: 1500,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false,
    }).start();

    // 3. Spring the card in.
    Animated.parallel([
      Animated.spring(entranceScale, {
        toValue: 1,
        friction: 6,
        tension: 100,
        useNativeDriver: true,
      }),
      Animated.timing(entranceOpacity, {
        toValue: 1,
        duration: 400,
        useNativeDriver: true,
      }),
    ]).start();

    return () => counterValue.removeListener(listener);
  }, [score]);

  // Pulse loop — only spun up for Megachopped so the screen isn't
  // generally distracting.
  useEffect(() => {
    if (!isMegaChopped) return;
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseScale, {
          toValue: 1.04,
          duration: 700,
          easing: Easing.inOut(Easing.quad),
          useNativeDriver: true,
        }),
        Animated.timing(pulseScale, {
          toValue: 1.0,
          duration: 700,
          easing: Easing.inOut(Easing.quad),
          useNativeDriver: true,
        }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [isMegaChopped]);

  // Card background: tinted version of the tier colour so it works in
  // both light and dark mode without clashing with the screen theme.
  // Hex → semi-transparent overlay; the tier colour shows through as
  // a soft tint instead of a saturated block.
  const cardTint = `${tier.color}1A`;   // ~10% alpha
  const borderTint = `${tier.color}80`; // ~50% alpha

  return (
    <Animated.View
      style={[
        styles.card,
        {
          backgroundColor: isDark ? `${tier.color}22` : cardTint,
          borderColor: borderTint,
          opacity: entranceOpacity,
          transform: [
            { scale: entranceScale },
            // Combining entrance scale with the pulse multiplier:
            // we can't directly multiply two Animated values inside
            // a single transform without explicit math, so we let
            // pulseScale ride on top via a separate `scaleY` and
            // accept a tiny visual artefact. Keeping it simple here:
            // pulse only kicks in once entrance has settled.
            { scaleY: isMegaChopped ? pulseScale : 1 },
          ],
        },
      ]}
    >
      {/* Top row: emoji + tier title */}
      <View style={styles.titleRow}>
        <Text style={[styles.emoji]}>{tier.emoji}</Text>
        <Text
          style={[
            styles.title,
            {
              color: tier.color,
              textShadowColor: isMegaChopped ? tier.glow : 'transparent',
              textShadowRadius: isMegaChopped ? 10 : 0,
            },
          ]}
        >
          {tier.label}
        </Text>
      </View>

      {/* Two-column body: score on the left, captions on the right */}
      <View style={styles.body}>
        <View style={styles.scoreColumn}>
          <Text style={[styles.scoreNumber, { color: tier.color }]}>
            {displayed}
          </Text>
          <Text style={[styles.scoreLabel, { color: isDark ? '#AAA' : '#666' }]}>
            chopped / 100
          </Text>
        </View>

        <View style={styles.captionColumn}>
          <Text
            style={[
              styles.caption,
              { color: isDark ? '#EEE' : '#222' },
            ]}
          >
            {caption}
          </Text>
          <Text
            style={[
              styles.subCaption,
              { color: isDark ? '#888' : '#777' },
            ]}
          >
            {100 - score >= 0 ? `${(100 - score).toFixed(1)} attractiveness` : ''}
          </Text>
        </View>
      </View>

      {/* Progress bar — visual representation of 0 → 100 */}
      <View
        style={[
          styles.progressTrack,
          { backgroundColor: isDark ? '#333' : '#E0E0E0' },
        ]}
      >
        <Animated.View
          style={[
            styles.progressFill,
            {
              backgroundColor: tier.color,
              width: counterValue.interpolate({
                inputRange: [0, 100],
                outputRange: ['0%', '100%'],
              }),
            },
          ]}
        />
      </View>

      <Text style={[styles.disclaimer, { color: isDark ? '#666' : '#999' }]}>
        Computed from facial attributes + a SCUT-FBP5500 beauty regressor.
        Subjective, culturally biased, and reductive. In-joke metric only.
      </Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  card: {
    marginHorizontal: 20,
    marginVertical: 16,
    padding: 20,
    borderRadius: 16,
    borderWidth: 1.5,
    overflow: 'hidden',
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  emoji: {
    fontSize: 30,
    marginRight: 10,
  },
  title: {
    fontSize: 28,
    fontWeight: '900',
    letterSpacing: 0.5,
  },
  body: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  scoreColumn: {
    alignItems: 'flex-start',
    marginRight: 18,
    minWidth: 90,
  },
  scoreNumber: {
    fontSize: 64,
    fontWeight: '900',
    lineHeight: 70,
    letterSpacing: -2,
    fontVariant: ['tabular-nums'],
  },
  scoreLabel: {
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 1,
    fontWeight: '600',
    marginTop: -4,
  },
  captionColumn: {
    flex: 1,
  },
  caption: {
    fontSize: 16,
    fontStyle: 'italic',
    fontWeight: '500',
    lineHeight: 22,
  },
  subCaption: {
    fontSize: 12,
    marginTop: 6,
    letterSpacing: 0.4,
  },
  progressTrack: {
    height: 8,
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 12,
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
  disclaimer: {
    fontSize: 10,
    fontStyle: 'italic',
    lineHeight: 14,
  },
});
