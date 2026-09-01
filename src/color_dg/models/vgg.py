"""VGG-11 fuer CIFAR-10 — zweite Architektur zur Absicherung der Befunde.

Warum VGG-11: Es ist keine Variante von ResNet, sondern ein rein
vorwaertsgerichtetes Netz ohne Residualverbindungen. Zugleich beginnt es —
wie das an CIFAR angepasste ResNet-18 — mit einer 3x3-Faltung auf drei
Eingangskanaelen, sodass die Analyse der Eingangsfilter unveraendert
uebertragbar ist. Mit norm="none" entfaellt jede Normalisierungsschicht;
damit laesst sich die in Abschnitt 8.2.1 formulierte Vorhersage pruefen.
"""

import torch.nn as nn

CFG11 = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class VGG11(nn.Module):
    def __init__(self, num_classes=10, norm="bn", gn_groups=32):
        super().__init__()
        layers, in_ch = [], 3
        for v in CFG11:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
                continue
            layers.append(nn.Conv2d(in_ch, v, 3, padding=1,
                                    bias=(norm == "none")))
            if norm == "bn":
                layers.append(nn.BatchNorm2d(v))
            elif norm == "gn":
                layers.append(nn.GroupNorm(min(gn_groups, v), v))
            elif norm != "none":
                raise ValueError(f"Unknown norm: {norm!r}")
            layers.append(nn.ReLU(inplace=True))
            in_ch = v
        self.features = nn.Sequential(*layers)
        # 32x32 -> fuenf Halbierungen -> 512 x 1 x 1
        self.classifier = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    @property
    def conv1(self):
        """Alias auf die erste Faltung — kein registriertes Untermodul,
        damit die Parameter nicht doppelt gezaehlt werden."""
        return self.features[0]

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def create_vgg11(num_classes=10, pretrained=False, norm="bn", gn_groups=32):
    if pretrained:
        raise ValueError("VGG-11 wird ausschliesslich von Grund auf trainiert.")
    return VGG11(num_classes=num_classes, norm=norm, gn_groups=gn_groups)
